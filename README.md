# MNIST_autoencoder

본 Repository 는 패스트캠퍼스의 딥러닝 초급 강좌와 AutoEncoder Paper를 참고로 제작되었습니다.  
Ref. <a href='https://arxiv.org/abs/2003.05991'>Autoencoder Paper</a>
---
## AutoEncoder란?

> An autoencoder is a specific type of a neural network, which is mainly designed to encode the input into a compressed and meaningful representation,
and then decode it back such that the reconstructed input is similar as possible as to the original one.  
-Autoencoder paper [Abstract]-

-> AutoEncoder는 뉴럴 네트워크의 구체적인 형태입니다, 오토인코더는 주로 입력을 압축하고 의미있는 표현으로 인코딩하고 그것을 디코딩작업을 하여 복원된 입력이 기존의 입력과 최대한 비슷하게 되돌리도록 디자인 되었습니다.

![IMG_9AB4D862AC3F-1](https://user-images.githubusercontent.com/76929568/214225284-53cf9750-4b19-41be-822e-20eb47248a79.jpeg)
- Encoder : 입력(input,x)의 정보를 최대한 보존하도록 손실압축을 수행
- Decoder : 중간 결과물(z)의 정보를 입력(input,x)과 같아지도록 압축 해제(복원)을 수행
- Bottleneck : 중간 결과물(z) 구간으로 입력(x)에 비해 작은 차원으로 구성된다. (이 과정에서 정보의 선택과 압축이 발생한다.-> 차원에 따라 압축정도가 결정)
> AutoEncoder는 압축과 해제를 반복하며 중요한 특징 추출을 자동으로 학습한다.  

---

> Bottleneck의 중간 결과물(z)는 입력(input,x)의 Feature Vector라고 볼 수 있고 입력(input,x)보다 차원수가 작기 때문에 입력에 비해 Dense Vector라고 볼 수 있다.


---
## Dataset - MNIST
- MNIST 숫자 손글씨 데이터 (0 ~ 9 Class , 28x28 이미지 데이터)
- MNIST 데이터 구성
  - 학습 데이터 : 60,000개
  - 테스트 데이터 : 10,000개

## Train/Valid Split
- Train : Valid = 8 : 2
- |Train| = (48,000 , 784)
- |Valid| = (12,000 , 784)

---
## Configuration
```
def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn',required=True)
    p.add_argument('--gpu_id',type=int,default=0 if torch.cuda.is_available() else -1)
    
    p.add_argument('--n_epochs',type=int,default=10)
    p.add_argument('--batch_size',type=int,default=256)
    p.add_argument('--train_ratio',type=float,default=.8)
    
    p.add_argument('--n_layers',type=int,default=5)
    p.add_argument('--btl_size',type=int,default=10)
    p.add_argument('--use_dropout',action='store_true')
    p.add_argument('--dropout_p',type=float,default=.3)

    p.add_argument('--verbose',type=int,default=1)

    config=p.parse_args()

    return config
```
---
## Model Architecture
Block
- Block 구성 : nn.Linear() + nn.ReLU() + Regularizer(Dropout or BatchNormalization)
```
class Block(nn.Module):
    def __init__(self,
                input_size,
                output_size,
                use_batch_norm=True,
                dropout_p=.4):
        self.input_size = input_size
        self.output_size= output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p=dropout_p
        
        super().__init__()

        def get_regularizer(use_batch_norm,size):
            if use_batch_norm:
                return nn.BatchNorm1d(size)
            else:
                return nn.Dropout(dropout_p)

        self.block = nn.Sequential(
            nn.Linear(input_size,output_size),
            nn.ReLU(),
            get_regularizer(use_batch_norm,output_size)
        )

    def forward(self,x):

        y=self.block(x)

        return y

```

AutoEncoder
- Encoder 와 Decoder 는 대칭 구조
```
class AutoEncoder(nn.Module):

    def __init__(self,
                input_size,
                output_size,
                hidden_sizes,
                btl_size=2,
                use_batch_norm=True,
                dropout_p=.3,
                ):

        assert len(hidden_sizes) > 0 ,"You need to specify hidden layers."

        super().__init__()

        last_hidden_size = input_size
        encoder_blocks=[]

        for hidden_size in hidden_sizes[1:]:
            encoder_blocks+=[Block(
                            last_hidden_size,
                            hidden_size,
                            use_batch_norm,
                            )]
            last_hidden_size = hidden_size

        self.encoder = nn.Sequential(
            *encoder_blocks,
            nn.Linear(last_hidden_size,btl_size),
        )

        decoder_blocks=[]
        last_hidden_size=btl_size

        for hidden_size in hidden_sizes[1::-1]:
            decoder_blocks+=[Block(
                last_hidden_size,
                hidden_size,
                use_batch_norm
                )]
            last_hidden_size=hidden_size
        
        self.decoder = nn.Sequential(
            *decoder_blocks,
            nn.Linear(last_hidden_size,input_size),
        )

    def forward(self,x):
        # |x| = (batch_size,input_size)
        # |z| = (batch_size,btl_size)
        z = self.encoder(x)

        # |y| = (batch_size,input_size)
        y = self.decoder(z)

        return y
```

---

## Train
- Loss Function : nn.MSELoss()
- Optimizer : optim.Adam()
```
python train.py [-h] --model_fn MODEL_FN [--gpu_id GPU_ID] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE] [--train_ratio TRAIN_RATIO] [--n_layers N_LAYERS] [--btl_size BTL_SIZE] [--use_dropout] [--dropout_p DROPOUT_P] [--verbose VERBOSE]
```

## 실험 평가
Configuration
- gpu_id : CPU (M1 in mac mini)
- n_epochs : 20
- batch_size : 256
- train_ratio : .8
- n_layers : 10
- btl_size : 2
- verbose : 1
- 
평가에 관련된 자세한 내용은 <a href='https://github.com/faizman31/MNIST_autoencoder/blob/main/predict.ipynb'>Predict.ipynb</a> 에서 보실수 있습니다.
