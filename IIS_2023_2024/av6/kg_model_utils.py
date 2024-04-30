import torch
from torch_geometric.nn import TransE, ComplEx


def train(model, data_loader, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_examples = 0

        for head_index, rel_type, tail_index in data_loader:
            optimizer.zero_grad()
            loss = model.loss(head_index, rel_type, tail_index)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * head_index.numel()
            total_examples += head_index.numel()

        loss = total_loss / total_examples
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


def evaluate(model, data_loader):
    hits1_list = []
    hits3_list = []
    hits10_list = []
    mr_list = []
    mrr_list = []

    for head_index, rel_type, tail_index in data_loader:
        head_embeds = model.node_emb(head_index)
        relation_embeds = model.rel_emb(rel_type)
        tail_embeds = model.node_emb(tail_index)

        if isinstance(model, TransE):
            scores = torch.norm(head_embeds + relation_embeds - tail_embeds, p=1, dim=1)

        elif isinstance(model, ComplEx):
            # Get real and imaginary parts
            re_relation, im_relation = torch.chunk(relation_embeds, 2, dim=1)
            re_head, im_head = torch.chunk(head_embeds, 2, dim=1)
            re_tail, im_tail = torch.chunk(tail_embeds, 2, dim=1)

            # Compute scores
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            scores = (re_score * re_tail + im_score * im_tail)

            # Negate as we want to rank scores in ascending order, lower the better
            scores = - scores.sum(dim=1)

        else:
            raise ValueError(f'Unsupported model.')

        scores = scores.view(-1, head_embeds.size()[0])

        hits1, hits3, hits10, mr, mrr = eval_metrics(scores)
        hits1_list.append(hits1.item())
        hits3_list.append(hits3.item())
        hits10_list.append(hits10.item())
        mr_list.append(mr.item())
        mrr_list.append(mrr.item())

    hits1 = sum(hits1_list) / len(hits1_list)
    hits3 = sum(hits3_list) / len(hits1_list)
    hits10 = sum(hits10_list) / len(hits1_list)
    mr = sum(mr_list) / len(hits1_list)
    mrr = sum(mrr_list) / len(hits1_list)

    return hits1, hits3, hits10, mr, mrr


def eval_metrics(y_pred):
    argsort = torch.argsort(y_pred, dim=1, descending=False)
    # not using argsort to do the rankings to avoid bias when the scores are equal
    ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
    ranking_list = ranking_list[:, 1] + 1
    hits1_list = (ranking_list <= 1).to(torch.float)
    hits3_list = (ranking_list <= 3).to(torch.float)
    hits10_list = (ranking_list <= 10).to(torch.float)
    mr_list = ranking_list.to(torch.float)
    mrr_list = 1. / ranking_list.to(torch.float)

    return hits1_list.mean(), hits3_list.mean(), hits10_list.mean(), mr_list.mean(), mrr_list.mean()
