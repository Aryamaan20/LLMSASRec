import os
import logging
import time
import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if str(device) == 'cuda':
    torch.cuda.set_device(0)  
    print(f"\n=== GPU ACTIVATED ===\n"
          f"Device: {torch.cuda.get_device_name(0)}\n"
          f"Memory: Allocated = {torch.cuda.memory_allocated()/1e9:.2f}GB, "
          f"Cached = {torch.cuda.memory_reserved()/1e9:.2f}GB\n")
else:
    print("\n=== USING CPU ===\n")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Args:
    dataset = "/kaggle/input/amazonbeauty/Beauty.txt"
    train_dir = "train_output"
    batch_size = 512
    lr = 0.001
    maxlen = 200
    hidden_units = 50
    num_blocks = 4
    num_epochs = 50
    num_heads = 2
    dropout_rate = 0.2
    l2_emb = 0.0
    weight_decay = 0.01
    warmup_steps = 4000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_only = False
    state_dict_path = None

if __name__ == '__main__':

    args = Args()
    
    os.makedirs(args.train_dir, exist_ok=True)
    logger.info(f"Created output directory at: {os.path.abspath(args.train_dir)}")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  
        
    dataset = data_partition(args.dataset)
    user_train, user_valid, user_test, usernum, itemnum = dataset
    
    model = SASRecWithBERT(usernum, itemnum, args).to(args.device)

    optimizer = torch.optim.AdamW([
        {'params': [p for n,p in model.named_parameters() if 'bert' not in n]},
        {'params': [p for n,p in model.named_parameters() if 'bert' in n], 'lr': 1e-5}
    ], lr=args.lr, weight_decay=args.weight_decay)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    logger.info("Creating sampler...")
    sampler = WarpSampler(user_train, usernum, itemnum, 
                         batch_size=args.batch_size,
                         maxlen=args.maxlen,
                         n_workers=4)
    
    scheduler = CosineAnnealingLR(optimizer, args.num_epochs)
    scaler = GradScaler()
    
    best_ndcg = 0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        with tqdm(total=len(user_train)//args.batch_size+1,
                 desc=f'Epoch {epoch}/{args.num_epochs}',
                 bar_format='{l_bar}{bar:20}{r_bar}',
                 mininterval=1) as pbar:
            
            for step in range(len(user_train) // args.batch_size + 1):
                batch = sampler.next_batch()
                u = torch.LongTensor(batch['uids']).to(args.device)
                seq = torch.LongTensor(batch['seq']).to(args.device)
                pos = torch.LongTensor(batch['pos']).to(args.device)
                neg = torch.LongTensor(batch['neg']).to(args.device)

                bert_inputs = {
                'input_ids': torch.LongTensor(batch['bert_input_ids']).to(device),
                'attention_mask': torch.LongTensor(batch['bert_attention_mask']).to(device)
                }
                       
                with autocast():
                    pos_logits, neg_logits = model(u, seq, pos, neg)
                    loss = torch.nn.BCEWithLogitsLoss()(
                        pos_logits[pos != 0], 
                        torch.ones_like(pos_logits[pos != 0])
                    )
                    loss += torch.nn.BCEWithLogitsLoss()(
                        neg_logits[pos != 0],
                        torch.zeros_like(neg_logits[pos != 0])
                    )
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                pbar.update(1)
        
        avg_loss = epoch_loss / (len(user_train)//args.batch_size + 1)
        logger.info(f"Epoch {epoch} complete - Loss: {avg_loss:.4f} - Time: {time.time()-start_time:.1f}s")
        
        if epoch % 5 == 0 or epoch == 1 or epoch==2 or epoch==3:
            val_ndcg, val_hr = evaluate_valid(model, dataset, args)
            logger.info(f"Validation - NDCG@10: {val_ndcg:.4f}, HR@10: {val_hr:.4f}")
            
            if val_ndcg > best_ndcg:
                best_ndcg = val_ndcg
                model_path = os.path.join(args.train_dir, f'best_model_epoch{epoch}.pth')
                torch.save(model.state_dict(), model_path)
                logger.info(f"Saved best model to: {model_path}")
    
    sampler.close()
