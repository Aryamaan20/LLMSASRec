import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_units, eps=1e-8)

    def forward(self, inputs):
        outputs = inputs.transpose(-1, -2)
        outputs = self.conv2(F.gelu(self.dropout1(self.conv1(outputs))))
        outputs = self.dropout2(outputs).transpose(-1, -2)
        return self.layer_norm(outputs + inputs)

class SASRecWithBERT(nn.Module):
    def __init__(self, num_users, num_items, args):
        super().__init__()
        self.num_items = num_items + 1
        self.num_users = num_users + 1
        self.hidden_units = args.hidden_units
        self.maxlen = args.maxlen
        self.device = args.device
        
        self.bert_model = AutoModel.from_pretrained('prajjwal1/bert-mini')
        self.bert_tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-mini')
        self.bert_tokenizer.add_special_tokens({'additional_special_tokens': [f'[ITEM_{i}]' for i in range(1, num_items + 1)]})
        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))
        
        for param in self.bert_model.parameters():
            param.requires_grad = False
            
        self.item_emb = nn.Embedding(self.num_items, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.bert_proj = nn.Linear(256, args.hidden_units)
        
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        
        for _ in range(args.num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(args.hidden_units))
            self.attention_layers.append(nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate))
            self.forward_layernorms.append(nn.LayerNorm(args.hidden_units))
            self.forward_layers.append(nn.Sequential(
                nn.Linear(args.hidden_units, args.hidden_units * 4),
                nn.GELU(),
                nn.Linear(args.hidden_units * 4, args.hidden_units),
                nn.Dropout(args.dropout_rate)
            ))
        
        self.dropout = nn.Dropout(args.dropout_rate)
        self.final_layer = nn.Linear(args.hidden_units, args.hidden_units)
        self.to(self.device)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.bert_proj.weight)
        for layer in self.forward_layers:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)

    def log2feats(self, log_seqs):
        
        log_seqs = log_seqs.to(self.device).long()

        log_seqs = torch.clamp(log_seqs, 0, self.num_items - 1)


        assert torch.all(log_seqs >= 0), "Negative values in input sequence"
        assert torch.all(log_seqs < self.num_items), f"Item IDs exceed maximum ({self.num_items-1})"
    
        seqs = self.item_emb(log_seqs) * math.sqrt(self.hidden_units)
        
        with torch.no_grad():
            seq_texts = []
        for seq in log_seqs.cpu():  
            items = []
            for x in seq:
                val = x.item()  
                if val != 0:    
                    items.append(f"[ITEM_{val}]")
            seq_text = " ".join(items) if items else "[PAD]"
            seq_texts.append(seq_text)

        try:
            inputs = self.bert_tokenizer(
                seq_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.maxlen
            ).to(self.device)
            
            bert_out = self.bert_model(**inputs).last_hidden_state
            bert_feats = self.bert_proj(bert_out)
        except Exception as e:
            print(f"Error in BERT processing: {e}")
            print(f"Problematic sequences: {seq_texts}")
            raise

        if bert_feats.size(1) < seqs.size(1):
            bert_feats = F.pad(bert_feats, (0, 0, 0, seqs.size(1) - bert_feats.size(1)))
        elif bert_feats.size(1) > seqs.size(1):
            bert_feats = bert_feats[:, :seqs.size(1), :]
        
        seqs += bert_feats
        positions = torch.arange(self.maxlen, dtype=torch.long, device=self.device).unsqueeze(0)
        seqs += self.pos_emb(positions)
        seqs = self.dropout(seqs)
        
        timeline_mask = log_seqs == 0
        seqs *= ~timeline_mask.unsqueeze(-1)
        
        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q.transpose(0, 1), Q.transpose(0, 1), Q.transpose(0, 1), key_padding_mask=timeline_mask)
            seqs = Q + mha_outputs.transpose(0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = seqs + self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)
        
        return self.final_layer(seqs)

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        pos_logits = (log_feats * self.item_emb(pos_seqs)).sum(dim=-1)
        neg_logits = (log_feats * self.item_emb(neg_seqs)).sum(dim=-1)
        return pos_logits, neg_logits

    @torch.no_grad()
    def predict(self, user_ids, seq_input, item_indices):
        log_feats = self.log2feats(seq_input if isinstance(seq_input, torch.Tensor) else seq_input['input_ids'])[:, -1, :]
        item_tensor = torch.as_tensor(item_indices, dtype=torch.long, device=self.device)
        item_embs = self.item_emb(item_tensor)
        return (log_feats.unsqueeze(1) @ item_embs.unsqueeze(-1)).squeeze(-1).squeeze(1)
