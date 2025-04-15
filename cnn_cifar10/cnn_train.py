import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def main():
    # GPU 강제 사용 설정
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            torch.cuda.current_device()
            print("CUDA 사용: ", torch.cuda.get_device_name(0))
        except Exception as e:
            print("CUDA 디바이스 확인 실패, CPU로 대체합니다.", e)
            device = torch.device("cpu")
    else:
        print("CUDA 사용 불가 - CPU로 전환됩니다.")
        device = torch.device("cpu")
    
    print("Using device:", device)

    # 데이터 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 데이터셋 다운로드
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)

    # CNN 모델 정의
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 64 * 8 * 8)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = CNNModel().to(device)
    print(f"Model is on device: {next(model.parameters()).device}")

    # 손실 함수, 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    # 학습
    num_epochs = 10
    train_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            if epoch == 0 and i == 0:
                print(f"Batch 0 - images device: {images.device}, labels device: {labels.device}")

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training Finished.")

    # 테스트 정확도 평가
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Loss 시각화
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("training_loss.png")
    plt.show()

if __name__ == "__main__":
    main()
