from hybridnmt import TransformerLSTM
import random
import os
from torch.profiler import profile, record_function, ProfilerActivity

import torch

OUT_DIR = "./.out/"

def gen_train_data(data_len):
    train_x_data = [random.sample(range(50), random.randint(5,15)) for i in range(data_len)]
    for x in train_x_data:
        random.shuffle(x)
    train_data = [(x,x[::-1]) for x in train_x_data]
    train_data = [(torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)) for x,y in train_data]
    return train_data

def validate(model, val_data):
    """
    Validate model against unseen data
    """
    criterion = torch.nn.CrossEntropyLoss()
    total_error = 0
    for x, y in val_data:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        y_len = y.size(1)
        y_pred = model.forward(x, y)
        error = 0
        for di in range(0, y_len):
            error += criterion(y_pred[:,di-1,:],y[:,di])
        error /= y_len
        total_error += error
    return (total_error / len(val_data)).item()

token_dict = {x:x for x in range(90)}
model = TransformerLSTM(token_dict)
if os.path.exists(OUT_DIR + 'sanity_check_model'):
    model = torch.load(OUT_DIR + 'sanity_check_model')
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
criterion = torch.nn.CrossEntropyLoss()
batch_size=4
num_epochs = 40

train_data = gen_train_data(2000)
val_data = gen_train_data(40)

for epoch in range(num_epochs):
    random.shuffle(train_data)
    print(f"epoch {epoch+1}/{num_epochs}")
    #with profile(activities=[ProfilerActivity.CPU], with_stack=True,
    #                profile_memory=True, record_shapes=True) as prof:
    for iteration in range(0, len(train_data), batch_size):
        src = torch.nn.utils.rnn.pad_sequence([train_data[i][0] for i in range(iteration, iteration+batch_size)], batch_first=True)
        tgt = torch.nn.utils.rnn.pad_sequence([train_data[i][1] for i in range(iteration, iteration+batch_size)], batch_first=True)
        loss = model.train(src, tgt, optimizer, criterion)
        del src
        del tgt
    val_loss = validate(model, val_data)
    train_loss = validate(model, train_data)
    print(f"VAL LOSS - {val_loss}")
    print(f"TRAIN LOSS - {train_loss}")

    #print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
torch.save(model, OUT_DIR + 'sanity_check_model')
