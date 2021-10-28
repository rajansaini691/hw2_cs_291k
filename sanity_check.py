from hybridnmt import TransformerLSTM
import random

import torch

train_x_data = [list(range(i,i+10)) for i in range(40)]
for x in train_x_data:
    random.shuffle(x)
train_data = [(x,x[::-1]) for x in train_x_data]
train_data = [(torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)) for x,y in train_data]

model = TransformerLSTM(vocab_size=50)
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(20):
    random.shuffle(train_data)
    print(f"epoch {epoch+1}/20")
    for iteration in range(len(train_data)):
        src = train_data[iteration][0].unsqueeze(0)
        tgt = train_data[iteration][1].unsqueeze(0)
        loss = model.train(src, tgt, optimizer, criterion)
        print(loss)
