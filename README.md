# picture-filling-using-GAN
Combines two discriminators and one generator to fill the missing part of a picture

pure tensorflow version of GAN(Generative Adversarial Networks)

Under my improvement, I add another discriminator as global_discriminator for better performance.

生成对抗神经网络纯tensorflow版本， 加了两个鉴别器共同作用于填充缺失的图片，反向传播的时候计算两个鉴别器共同的损失，然后进行反向传播
有一点点类似于特征金字塔的思想，多尺度预测。
