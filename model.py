import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F

class OneHeadAttention( nn.Module ):

    def __init__( self, e_dim, h_dim ):
        '''
        :param e_dim:
        :param h_dim:
        '''
        super().__init__()
        self.h_dim = h_dim

        self.lQ = nn.Linear( e_dim, h_dim )
        self.lK = nn.Linear( e_dim, h_dim )
        self.lV = nn.Linear( e_dim, h_dim )

    def forward( self, seq_inputs , querys = None, mask = None ):
        '''
        :param seq_inputs: #[ batch, seq_lens, e_dim ]
        :param querys: #[ batch, seq_lens, e_dim ]
        :param mask: #[ 1, seq_lens, seq_lens ]
        :return:
        '''
        if querys is not None:
            Q = self.lQ( querys ) #[ batch, seq_lens, h_dim ]
        else:
            Q =  self.lQ( seq_inputs ) #[ batch, seq_lens, h_dim ]
        K = self.lK( seq_inputs ) #[ batch, seq_lens, h_dim ]
        V = self.lV( seq_inputs ) #[ batch, seq_lens, h_dim ]
        # [ batch, seq_lens, seq_lens ]
        QK = torch.matmul( Q,K.permute( 0, 2, 1 ) )
        # [ batch, seq_lens, seq_lens ]
        QK /= ( self.h_dim ** 0.5 )

        if mask is not None:
            QK = QK.masked_fill( mask == 0, -1e9 )
        # [ batch, seq_lens, seq_lens ]
        a = torch.softmax( QK, dim = -1 )
        # [ batch, seq_lens, h_dim ]
        outs = torch.matmul( a, V ) # (batch_size,seq_len, embed_size)
        # (batch_size, embed_size, seq_len)
        pooled_output = F.adaptive_avg_pool1d(outs.permute(0, 2, 1), 1).squeeze(2)

        return pooled_output # (batch_size, embed_size)



class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        nn.init.normal_(self.query.weight, std=0.1)
        nn.init.normal_(self.key.weight, std=0.1)
        nn.init.normal_(self.value.weight, std=0.1)

    def forward(self, x):

        batch_size = x.size(0)

        q = self.query(x).view(batch_size, -1, 1)  # [batch_size, input_dim, 1]
        k = self.key(x).view(batch_size, 1, -1)  # [batch_size, 1, input_dim]
        v = self.value(x).view(batch_size, 1, -1)  # [batch_size, 1, input_dim]

        scores = torch.matmul(q, k)  # [batch_size, input_dim, input_dim]
        weights = F.softmax(scores.squeeze(1), dim=1).unsqueeze(1)  # [batch_size, 1, input_dim]
        output = torch.matmul(weights, v).squeeze(1)  # [batch_size, input_dim]

        return output


class SelfAttention2(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention2, self).__init__()
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, v)

        return output
class Net(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_heads):
        super(Net, self).__init__()

        self.num_users = num_users
        self.num_items =num_items
        self.total_nodes = num_users + num_items

        # node embedding layer,include all users and items
        self.user_item_embedding = nn.Embedding(self.total_nodes, embedding_dim)
        self.all_user_item_indexes = torch.LongTensor(range(self.total_nodes))

        # user implicit relation graph
        self.gat_conv_user = GATConv(embedding_dim, embedding_dim, heads=num_heads)
        # item implicit relation graph
        self.gat_conv_item = GATConv(embedding_dim, embedding_dim, heads=num_heads)
        # user-item interaction graph
        self.gat_conv_user_item = GATConv(embedding_dim, embedding_dim, heads=num_heads)

        self.mlp_user = nn.Sequential(
            nn.Linear(embedding_dim * num_heads, embedding_dim),
            nn.ReLU(),
        )
        self.mlp_item = nn.Sequential(
            nn.Linear(embedding_dim * num_heads, embedding_dim),
            nn.ReLU(),
        )

        self.attention_user = nn.Sequential(
            nn.Linear(embedding_dim * num_heads, embedding_dim),
            nn.ReLU(),
            SelfAttention(embedding_dim)
        )
        self.attention_item = nn.Sequential(
            nn.Linear(embedding_dim * num_heads, embedding_dim),
            nn.ReLU(),
            SelfAttention(embedding_dim)
        )

        self.attention_user_aggr=OneHeadAttention(embedding_dim,embedding_dim)
        self.attention_item_aggr=OneHeadAttention(embedding_dim,embedding_dim)


        self.mlp_final = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, user_indices, item_indices, edge_index_user, edge_index_item,edge_index_user_item):

        x_user_item = self.user_item_embedding( self.all_user_item_indexes )
        x_user=x_user_item[:self.num_users,:]
        x_item=x_user_item[self.num_users:,:]

        x_user = self.gat_conv_user(x_user, edge_index_user)
        x_user=x_user[user_indices]
        x_user=self.mlp_user(x_user)

        x_item = self.gat_conv_item(x_item, edge_index_item)
        x_item=x_item[item_indices]
        x_item=self.mlp_item(x_item)

        x_user_item = self.gat_conv_user_item(x_user_item,edge_index_user_item)

        x_user_ui=x_user_item[:self.num_users,:]

        x_item_ui=x_user_item[self.num_users:,:]

        x_user_ui=x_user_ui[user_indices]

        x_item_ui=x_item_ui[item_indices]

        x_user_ui=self.mlp_user(x_user_ui)
        x_item_ui=self.mlp_item(x_item_ui)

        attended_user_rep = torch.cat((x_user.unsqueeze(1), x_user_ui.unsqueeze(1)), dim=1)
        attended_user_rep=self.attention_user_aggr(attended_user_rep)

        attended_item_rep = torch.cat((x_item.unsqueeze(1), x_item_ui.unsqueeze(1)), dim=1)
        attended_item_rep=self.attention_item_aggr(attended_item_rep)

        combined_rep = torch.cat([attended_user_rep, attended_item_rep], dim=1)
        interaction_prob = torch.sigmoid(self.mlp_final(combined_rep))

        return interaction_prob