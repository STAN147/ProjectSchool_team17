import torch
import torch.nn.functional as F

def sample_candidates(train_size: int, batch_indices: torch.Tensor, sample_rate: float) -> torch.Tensor:
    available = torch.arange(train_size)
    mask = ~torch.isin(available, batch_indices)
    available = available[mask]
    if available.numel() < 2:
        raise ValueError("At least two candidate examples are required")
    num_samples = min(available.numel(), max(2, int(train_size * sample_rate)))
    shuffled = torch.randperm(available.size(0))
    return available[shuffled[:num_samples]]

def compute_neighbor_probabilities(query_batch: torch.Tensor, sample_candidates_batch: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    distances = torch.cdist(query_batch, sample_candidates_batch, p=2.0)
    distances = -distances / T if T != 1.0 else -distances
    probabilities = F.softmax(distances, dim=-1)
    return probabilities

def compute_nca_predictions(probabilities: torch.Tensor, sample_candidates_y: torch.Tensor, task_type: str, num_classes: int = None) -> torch.Tensor:
    if task_type in ("binary", "multiclass"):
        y_one_hot = F.one_hot(sample_candidates_y.long().view(-1), num_classes=num_classes).float()
        predictions = probabilities @ y_one_hot
    if task_type == "regression":
        y = sample_candidates_y.view(-1, 1).float()
        predictions = (probabilities @ y).squeeze(-1)
    return predictions

def compute_nca_loss(query_batch_predictions, query_batch_y, task_type: str) -> torch.Tensor:
    if task_type in ("binary", "multiclass"):
        eps = 1e-8
        log_probs = torch.log(query_batch_predictions + eps)
        loss = F.nll_loss(log_probs, query_batch_y.long().view(-1))
    if task_type == "regression":
        loss = F.mse_loss(query_batch_predictions, query_batch_y.float().view(-1))
    return loss

class Trainer:
    def __init__(self, encoder: torch.nn.Module, config: dict):
        self.encoder = encoder
        self.learning_rate = config.get("lr", 1e-3)
        self.batch_size = config.get("batch_size", 1024)
        self.optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.learning_rate, 
            weight_decay=config.get("weight_decay", 1e-4))

    def fit(self, train_data: tuple, epochs: int, sample_rate: float, task_type: str, num_classes: int = None, val_data: tuple = None):
        x_num_train, x_cat_train, y_train = train_data
        print(x_cat_train)
        train_size = y_train.size(0)
        if train_size < 4:
            raise ValueError("At least four training examples are required for BatchNorm/SNS")
        batch_size = min(self.batch_size, train_size - 2)
        for epoch in range(epochs):
            self.encoder.train()
            s = torch.randperm(train_size)
            for i in range(0, train_size, batch_size):
                batch_indices = s[i:i + batch_size]
                if batch_indices.numel() < 2:
                    continue
                query_x_num = x_num_train[batch_indices] if x_num_train is not None else None
                query_x_cat = x_cat_train[batch_indices] if x_cat_train is not None else None
                query_y = y_train[batch_indices]

                candidate_indices = sample_candidates(train_size, batch_indices, sample_rate)
                cand_x_num = x_num_train[candidate_indices] if x_num_train is not None else None
                cand_x_cat = x_cat_train[candidate_indices] if x_cat_train is not None else None
                cand_y = y_train[candidate_indices]
                self.optimizer.zero_grad()

                z_query = self.encoder(query_x_num, query_x_cat)
                z_cand = self.encoder(cand_x_num, cand_x_cat)

                probs = compute_neighbor_probabilities(z_query, z_cand)
                predictions = compute_nca_predictions(probs, cand_y, task_type, num_classes)
                loss = compute_nca_loss(predictions, query_y, task_type)

                loss.backward()
                self.optimizer.step()

            if val_data is not None:
                val_loss = self.evaluate(train_data, val_data, task_type, num_classes)
                print(f"Epoch {epoch+1} val Loss: {val_loss}")

    def predict(self, train_reference_data: tuple, test_data: tuple, task_type: str, num_classes: int = None, max_candidates: int = 1000000) -> torch.Tensor:
        x_num_ref, x_cat_ref, y_ref = train_reference_data
        x_num_test = test_data[0]
        x_cat_test = test_data[1]
        
        ref_size = y_ref.size(0)
        test_size = (x_num_test if x_num_test is not None else x_cat_test).size(0)
        
        self.encoder.eval()
        all_predictions = []
        
        with torch.no_grad():
            ref_sample_size = min(max_candidates, ref_size)
            ref_idx = torch.randperm(ref_size)[:ref_sample_size]
            
            cand_y = y_ref[ref_idx]
            z_cand = self.encoder(
                x_num_ref[ref_idx] if x_num_ref is not None else None,
                x_cat_ref[ref_idx] if x_cat_ref is not None else None,
            )
            
            for i in range(0, test_size, self.batch_size):
                query_x_num = x_num_test[i : i + self.batch_size] if x_num_test is not None else None
                query_x_cat = x_cat_test[i : i + self.batch_size] if x_cat_test is not None else None
                z_query = self.encoder(query_x_num, query_x_cat)
                probabilities = compute_neighbor_probabilities(z_query, z_cand, T=1.0)
                predictions = compute_nca_predictions(probabilities, cand_y, task_type, num_classes)
                all_predictions.append(predictions)

        return torch.cat(all_predictions, dim=0)

    def evaluate(self, train_reference_data: tuple, val_data: tuple, task_type: str, num_classes: int = None, max_candidates: int = 100000) -> float:
        x_num_ref, x_cat_ref, y_ref = train_reference_data
        x_num_val, x_cat_val, y_val = val_data
        
        ref_size = y_ref.size(0)
        val_size = y_val.size(0)
        
        self.encoder.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            ref_sample_size = min(max_candidates, ref_size)
            ref_idx = torch.randperm(ref_size)[:ref_sample_size]
            
            cand_y = y_ref[ref_idx]
            z_cand = self.encoder(
                x_num_ref[ref_idx] if x_num_ref is not None else None,
                x_cat_ref[ref_idx] if x_cat_ref is not None else None,
            )

            for i in range(0, val_size, self.batch_size):
                query_x_num = x_num_val[i : i + self.batch_size] if x_num_val is not None else None
                query_x_cat = x_cat_val[i : i + self.batch_size] if x_cat_val is not None else None
                query_y = y_val[i : i + self.batch_size]
                
                z_query = self.encoder(query_x_num, query_x_cat)
                
                probabilities = compute_neighbor_probabilities(z_query, z_cand, T=1.0)
                predictions = compute_nca_predictions(probabilities, cand_y, task_type, num_classes)
                loss = compute_nca_loss(predictions, query_y, task_type)
                
                total_loss += loss.item() * query_y.shape[0]

        return total_loss / val_size
