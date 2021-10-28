from hybridnmt import TransformerLSTM
import random
from torch.profiler import profile, record_function, ProfilerActivity

import torch

train_x_data = [list(range(i,i+random.randint(4,14))) for i in range(40)]
for x in train_x_data:
    random.shuffle(x)
train_data = [(x,x[::-1]) for x in train_x_data]
train_data = [(torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)) for x,y in train_data]

model = TransformerLSTM(vocab_size=55)
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()
batch_size=4
num_epochs = 80

for epoch in range(num_epochs):
    random.shuffle(train_data)
    print(f"epoch {epoch+1}/{num_epochs}")
    with profile(activities=[ProfilerActivity.CPU], with_stack=True,
                    profile_memory=True, record_shapes=True) as prof:
        for iteration in range(0, len(train_data), batch_size):
            src = torch.nn.utils.rnn.pad_sequence([train_data[i][0] for i in range(iteration, iteration+batch_size)], batch_first=True)
            tgt = torch.nn.utils.rnn.pad_sequence([train_data[i][1] for i in range(iteration, iteration+batch_size)], batch_first=True)
            loss = model.train(src, tgt, optimizer, criterion)
            del src
            del tgt
            print(loss)

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
