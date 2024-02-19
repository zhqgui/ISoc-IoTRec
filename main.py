import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from preprocess import load_data,graphSampling
from model import Net
import numpy as np
from evaluation_func import hr_ndcg

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1M', help='dataset name: ml-1M/Ciao/Epinions')
parser.add_argument('--validation_prop', default=0.15, help='the proportion of data used for validation')
parser.add_argument('--neighbors_num', default=20, help='the number of neighbors for user or item')
parser.add_argument('--num_heads', default=1, help='the head number of Multi head Attention layer')
parser.add_argument('--lr', default=0.001, help='learning rate')
parser.add_argument('--batchSize',default=1024, help='batch size')
parser.add_argument('--epochs', default=20, help='number of epochs')
parser.add_argument('--hidden_dim',default=64, help='hidden dimension')

args = parser.parse_args()

@torch.no_grad()
def evaluate_model(net,data,user_graph,item_graph,user_item_graph,num_users,top_k=10,batchSize=args.batchSize,):
    net.eval()

    data_loader = DataLoader(data, batch_size=batchSize, shuffle=False)
    total_loss = 0.0
    total_hr=0.0
    total_ndcg=0.0
    criterion = torch.nn.BCELoss()
    for u, i, r in tqdm(data_loader):
        index_user = u.detach().numpy()
        index_item = i.detach().numpy()

        ui = torch.cat([u, i + num_users])
        index_user_item = ui.detach().numpy()
        r_float = torch.FloatTensor(r.detach().numpy())
        r_int = r.detach().numpy()
        edge_index_user = graphSampling(user_graph, index_user)
        edge_index_item = graphSampling(item_graph, index_item)
        edge_index_user_item = graphSampling(user_item_graph, index_user_item)
        logits = net(u, i, edge_index_user, edge_index_item, edge_index_user_item)

        loss = criterion(logits.squeeze(dim=1), r_float)

        predictions = logits.numpy().squeeze()

        all_predictions=np.column_stack((index_user, index_item, predictions))
        all_ground_truth=np.column_stack((index_user, index_item, r_int))

        all_predictions=np.array(all_predictions)
        all_ground_truth=np.array(all_ground_truth)

        hr, ndcg = hr_ndcg(all_ground_truth, all_predictions, 10)
        total_loss += loss
        total_hr+=hr
        total_ndcg+=ndcg

    average_loss = total_loss / len(data_loader)
    average_hr = total_hr / len(data_loader)
    average_ndcg = total_ndcg / len(data_loader)


    return average_loss,average_hr,average_ndcg
def train(epoch=args.epochs,batchSize=args.batchSize,dim=args.hidden_dim,lr=args.lr,validation_prop=args.validation_prop):
    dataset=args.dataset
    neighbors_num=args.neighbors_num
    num_heads=args.num_heads
    train_data, val_data,test_data,user_set, item_set, user_item_graph, user_graph, item_graph=load_data(dataset,neighbors_num,validation_prop)
    num_users=max(user_set)+1
    num_items=max(item_set)+1

    net = Net( num_users, num_items, dim,num_heads )
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    patience = 5
    best_val_loss = float('inf')
    print("Start training......")
    for e in range(epoch):
        net.train()
        all_lose = 0
        batch_count = 0
        train_loader = DataLoader(train_data, batch_size=batchSize, shuffle=True)

        for u, i, r in tqdm( train_loader ):
            index_user=u.detach().numpy()
            index_item=i.detach().numpy()

            ui=torch.cat([u,i+num_users])
            index_user_item=ui.detach().numpy()
            r = torch.FloatTensor( r.detach( ).numpy( ) )

            edge_index_user = graphSampling( user_graph, index_user )
            edge_index_item = graphSampling( item_graph, index_item )
            edge_index_user_item = graphSampling( user_item_graph, index_user_item )

            optimizer.zero_grad( )
            logits = net( u, i, edge_index_user, edge_index_item,edge_index_user_item )
            loss = criterion( logits.squeeze(dim=1), r )
            all_lose += loss
            loss.backward( )
            optimizer.step( )
            batch_count += 1

        average_train_loss=all_lose / len( train_loader )
        val_loss,hr,ndcg=evaluate_model(net,val_data,user_graph,item_graph,user_item_graph,num_users)
        print(f'\nEpoch {e + 1}/{epoch}, Average Train Loss: {average_train_loss}, Validation Loss: {val_loss}, HR: {hr}, NDCG: {ndcg}')

        # is early stopping?
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 5
        else:
            patience -= 1

        if patience == 0:
            print("Early stopping: No improvement in validation loss.")
            break

    # test
    test_loss,hr,ndcg =evaluate_model(net,test_data,user_graph,item_graph,user_item_graph,num_users)
    print(f'test results: HR: {hr}, NDCG: {ndcg}')
    torch.save(net, 'results/model.pth')
    torch.save(net.state_dict(), 'results/model_params.pth')

if __name__ == '__main__':
    train()
