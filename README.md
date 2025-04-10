## CNN 모형 만들기
CIFAR-10은 Canadian Institute for Advanced Research, 10 classes의 약자로, 10개의 카테고리로 분류되어 있는 6만개의 컬러 이미지를 자료이다. 이미지의 특징으로는
- 32 $\times$ 32 픽셀의 컬러 이미지
- 10개의 카테고리는 비행기(airplanes), 자동차(cars), 새(birds), 고양이(cats), 사슴(deer), 개(dogs), 개구리(frogs), 말(horses), 배(ships), 그리고 트럭(trucks) 이다.
- 각 카테고리는 6000개의 사진을 가지고 있다.
- 자료는 5만개의 학습용 이미지(training images)와 만개의 테스트 이미지(test images)로 구성된다.

CIFAR-10 자료를 이용해서 다음의 설명에 맞는 cnn 모형을 만들고 학습하는 과정을 확인하고 학습한 모형의 성능을 확인한다.

1. PyTorch 라이브러리에서 사용할 것을 import 한다.
2. CIFAR-10 자료를 사용하는 모형을 만들 때, 자료는 정규화(normalization) 해준다.
  - transforms를 torchvision 에서 import
  - ToTensor()를 이용해서 tensor 형태로 자료 변경 (0 ~ 1 사이의 값으로 변경, 화소값을 255로 나눔)
  - Normalize 함수를 사용해서 자료 값이 -1 ~ 1 사이의 값을 갖도록 변경
3. CIFAR-10자료를 이용하여 다음의 절차를 지키는 CNN 모형을 생성한다.
  - 학습용 배치 크기는 64로 설정
  - CNN 모델은 다음 조건을 따름
   - Conv($3 \times 3$, stride =1, padding=1, 32채널) → ReLU → MaxPool ($2 \times 2$, stride=2) → Conv($3 \times 3$, stride =1, padding=1, 64채널) → ReLU → MaxPool ($2 \times 2$, stride=2)
  - Fully Connected Layer는 128 노드, 최종 출력은 10 클래스, 활성화 함수는 ReLU를 사용

4. 손실함수는 `CrossEntropyLoss`, 옵티마이저는 `Adam`, 학습률은 0.0002
5. 총 10 epoch 동안 학습하며, 각 epoch마다 loss 값을 저장하고 그래프로 나타낸다.
6. 테스트 셋을 이용해 최종 정확도를 계산하고 출력

## 추가: GoogLeNet(Inception) 모델 구현

CIFAR-10 데이터셋을 사용하여 GoogLeNet 구조(Inception 모듈 기반)를 다음 조건에 맞게 구현하고 성능을 확인

1. 이미지 크기를 `transforms.Resize((224, 224))`로 변환한 후 `ToTensor` 및 `Normalize` 적용
2. GoogLeNet 구조를 직접 구현할 것 (Inception Module 포함)
3. 기본 채널 수(base_dim)는 64로 설정
4. 손실함수는 `CrossEntropyLoss`, 옵티마이저는 `Adam`, 학습률은 0.0002
5. 10 epoch 학습 후 테스트 정확도를 출력