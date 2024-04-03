import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("Llama")
model = AutoModel.from_pretrained("Llama")

inputs = tokenizer("这是一个例子", return_tensors="pt")
num_labels = 2
labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # 假设你的样本标签是这样的序列

classifier = torch.nn.Linear(
    model.config.hidden_size, num_labels
)  # 假设num_labels是我们任务的类别数量

outputs = model(**inputs)
logits = classifier(outputs.last_hidden_state[:, 0])
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    [
        {"params": model.parameters(), "lr": 1e-5},
        {"params": classifier.parameters(), "lr": 1e-3},
    ]
)
loss = loss_fn(logits, labels)  # 假设labels是我们的标签数据
loss.backward()
optimizer.step()
