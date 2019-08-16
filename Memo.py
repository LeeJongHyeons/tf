'''
Adam(확률적 최적화)
-Adaptive Moment Estimation을 지칭하고, 이전 Step에서 평균뿐만 아니라 분산까지 고려한 복잡한 지수 감쇠를 사용

SGD: Stochatic Gradient Descent를 지칭 
-Learning Rate
-Momentum: Local Mimima에서 빠지지 않기 위해 이전 단계에서의 가중치가 적용된 평균을 사용
-Nesterov Momentum: Solution에 가까워 질 수록 gradient를 Slow Down 시킴

RMSprop: Root Mean Squeared Error
- 지수 감쇠 Squared Gradient의 평균으로 나눔으로 Learning Rate를 감소

Adagrad: 매개 변수 별 학습 속도를 갖춘 최적화 프로그램으로 훈련 중 매개 변수가 얼마나 자주 업데이트 되는지 따라 조정

=================================================================================================================

Tensorflow
- Tensor 방식의 그래프로 계산법을 이용
- 형태 비교
  - 1 스칼라
  - (1, 2) 벡터 M * n 형태로 표현 => input_dim = 1 또는 input_shape=(1, )
  - [[1, 2]] 행렬

그래프에 대한 정의
y = wx+b
[y] [w] [x] [b]
[y] = 그래프 => (Marcine input) => (Session()) => (Sess.run) => (output) 
[w]*[x] = 곱하기 한 노드
[w]*[x] + [b] = add 값


'''