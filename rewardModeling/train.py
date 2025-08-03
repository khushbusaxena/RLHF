from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from data.dataset import HHRLHFPreferenceDataset
#from utils.tokenizer import collate_fn
from models.reward_model import RewardModelTrainer
from config import Config
import torch
#from transformers import AdamW
from torch.optim import AdamW

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_CHECKPOINT, num_labels=1)
model.to(Config.DEVICE)

# Load dataset
dataset = HHRLHFPreferenceDataset(split="train")
dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

# Initialize trainer
optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
loss_fn = torch.nn.BCEWithLogitsLoss()
trainer = RewardModelTrainer(model, Config.DEVICE, optimizer, loss_fn)

# Train the model
trainer.train(dataloader, Config.EPOCHS)
