#import torch
#from torch.utils.data import DataLoader
#from transformers import AutoModelForSequenceClassification, AdamW
import time

class RewardModelTrainer:
    def __init__(self, model, device, optimizer, loss_fn):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, dataloader, epochs=3):
        self.model.train()
        for epoch in range(epochs):

            epoch_loss = 0.0
            start_time = time.time()

            for step, batch in enumerate(dataloader):
                # Move tensors to the correct device
                preferred_input_ids = batch["preferred_input_ids"].to(self.device)
                preferred_attention_mask = batch["preferred_attention_mask"].to(self.device)
                non_preferred_input_ids = batch["non_preferred_input_ids"].to(self.device)
                non_preferred_attention_mask = batch["non_preferred_attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # Get rewards
                preferred_rewards = self.model(input_ids=preferred_input_ids, attention_mask=preferred_attention_mask).logits
                non_preferred_rewards = self.model(input_ids=non_preferred_input_ids, attention_mask=non_preferred_attention_mask).logits

                # Compute loss
                logits = preferred_rewards - non_preferred_rewards
                logits = logits.squeeze(-1)
                labels = labels.float()
                loss = self.loss_fn(logits, labels)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                if (step + 1) % 10 == 0 or (step + 1) == len(dataloader):
                    print(f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(dataloader)}] "
                          f"Loss: {loss.item():.4f}")

            epoch_time = time.time() - start_time
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}] completed in {epoch_time:.2f}s - Avg Loss: {avg_loss:.4f}\n")
