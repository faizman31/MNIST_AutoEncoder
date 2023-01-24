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
