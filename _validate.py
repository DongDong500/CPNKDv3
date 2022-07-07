import torch
import torch.nn as nn


def _validate(opts, s_model, t_model, loader, device, metrics, epoch, criterion):
    s_model.eval()
    metrics.reset()

    running_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):

            images = images.to(device)
            labels = labels.to(device)

            s_outputs = s_model(images)
            probs = nn.Softmax(dim=1)(s_outputs)
            preds = torch.max(probs, 1)[1].detach().cpu().numpy()
            target = labels.detach().cpu().numpy()

            t_outputs = t_model(images)

            loss = criterion(s_outputs, t_outputs, labels)

            metrics.update(target, preds)
            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    score = metrics.get_results()

    return score, epoch_loss