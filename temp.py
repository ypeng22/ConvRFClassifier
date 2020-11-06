correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = torch.tensor(images).to(dev)
        labels = torch.tensor(labels).to(dev)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.view(-1)).sum().item()
accuracy = float(correct) / float(total)