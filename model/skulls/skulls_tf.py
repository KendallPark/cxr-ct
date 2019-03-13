from kaffe.tensorflow import Network

class hourglass256x128x128_deploy(Network):
    def setup(self):
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, biased=False, relu=False, name='conv1_b')
             .batch_normalization(relu=True, name='bn_conv1_b')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res1_branch1')
             .batch_normalization(name='bn1_branch1'))

        (self.feed('bn_conv1_b')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res1_branch2a')
             .batch_normalization(relu=True, name='bn1_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res1_branch2b')
             .batch_normalization(relu=True, name='bn1_branch2b')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res1_branch2c')
             .batch_normalization(name='bn1_branch2c'))

        (self.feed('bn1_branch1', 
                   'bn1_branch2c')
             .add(name='res1')
             .relu(name='res1_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res4_branch2a')
             .batch_normalization(relu=True, name='bn4_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res4_branch2b')
             .batch_normalization(relu=True, name='bn4_branch2b')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res4_branch2c')
             .batch_normalization(name='bn4_branch2c'))

        (self.feed('res1_relu', 
                   'bn4_branch2c')
             .add(name='res4')
             .relu(name='res4_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res5_branch2a')
             .batch_normalization(relu=True, name='bn5_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res5_branch2b')
             .batch_normalization(relu=True, name='bn5_branch2b')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res5_branch2c')
             .batch_normalization(name='bn5_branch2c'))

        (self.feed('res4_relu', 
                   'bn5_branch2c')
             .add(name='res5')
             .relu(name='res5_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res6_branch1')
             .batch_normalization(name='bn6_branch1'))

        (self.feed('res5_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res6_branch2a')
             .batch_normalization(relu=True, name='bn6_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res6_branch2b')
             .batch_normalization(relu=True, name='bn6_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res6_branch2c')
             .batch_normalization(name='bn6_branch2c'))

        (self.feed('bn6_branch1', 
                   'bn6_branch2c')
             .add(name='res6')
             .relu(name='res6_relu')
             .max_pool(2, 2, 2, 2, name='hg1_pool1')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='reshg1_low1_branch2a')
             .batch_normalization(relu=True, name='bnhg1_low1_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='reshg1_low1_branch2b')
             .batch_normalization(relu=True, name='bnhg1_low1_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='reshg1_low1_branch2c')
             .batch_normalization(name='bnhg1_low1_branch2c'))

        (self.feed('hg1_pool1', 
                   'bnhg1_low1_branch2c')
             .add(name='reshg1_low1')
             .relu(name='reshg1_low1_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='reshg1_low2_branch2a')
             .batch_normalization(relu=True, name='bnhg1_low2_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='reshg1_low2_branch2b')
             .batch_normalization(relu=True, name='bnhg1_low2_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='reshg1_low2_branch2c')
             .batch_normalization(name='bnhg1_low2_branch2c'))

        (self.feed('reshg1_low1_relu', 
                   'bnhg1_low2_branch2c')
             .add(name='reshg1_low2')
             .relu(name='reshg1_low2_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='reshg1_low5_branch2a')
             .batch_normalization(relu=True, name='bnhg1_low5_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='reshg1_low5_branch2b')
             .batch_normalization(relu=True, name='bnhg1_low5_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='reshg1_low5_branch2c')
             .batch_normalization(name='bnhg1_low5_branch2c'))

        (self.feed('reshg1_low2_relu', 
                   'bnhg1_low5_branch2c')
             .add(name='reshg1_low5')
             .relu(name='reshg1_low5_relu')
             .max_pool(2, 2, 2, 2, name='hg1_low6_pool1')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low1_branch2a')
             .batch_normalization(relu=True, name='bnhg1_low6_low1_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low1_branch2b')
             .batch_normalization(relu=True, name='bnhg1_low6_low1_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='reshg1_low6_low1_branch2c')
             .batch_normalization(name='bnhg1_low6_low1_branch2c'))

        (self.feed('hg1_low6_pool1', 
                   'bnhg1_low6_low1_branch2c')
             .add(name='reshg1_low6_low1')
             .relu(name='reshg1_low6_low1_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low2_branch2a')
             .batch_normalization(relu=True, name='bnhg1_low6_low2_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low2_branch2b')
             .batch_normalization(relu=True, name='bnhg1_low6_low2_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='reshg1_low6_low2_branch2c')
             .batch_normalization(name='bnhg1_low6_low2_branch2c'))

        (self.feed('reshg1_low6_low1_relu', 
                   'bnhg1_low6_low2_branch2c')
             .add(name='reshg1_low6_low2')
             .relu(name='reshg1_low6_low2_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low5_branch2a')
             .batch_normalization(relu=True, name='bnhg1_low6_low5_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low5_branch2b')
             .batch_normalization(relu=True, name='bnhg1_low6_low5_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='reshg1_low6_low5_branch2c')
             .batch_normalization(name='bnhg1_low6_low5_branch2c'))

        (self.feed('reshg1_low6_low2_relu', 
                   'bnhg1_low6_low5_branch2c')
             .add(name='reshg1_low6_low5')
             .relu(name='reshg1_low6_low5_relu')
             .max_pool(2, 2, 2, 2, name='hg1_low6_low6_pool1')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low1_branch2a')
             .batch_normalization(relu=True, name='bnhg1_low6_low6_low1_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low1_branch2b')
             .batch_normalization(relu=True, name='bnhg1_low6_low6_low1_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low1_branch2c')
             .batch_normalization(name='bnhg1_low6_low6_low1_branch2c'))

        (self.feed('hg1_low6_low6_pool1', 
                   'bnhg1_low6_low6_low1_branch2c')
             .add(name='reshg1_low6_low6_low1')
             .relu(name='reshg1_low6_low6_low1_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low2_branch2a')
             .batch_normalization(relu=True, name='bnhg1_low6_low6_low2_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low2_branch2b')
             .batch_normalization(relu=True, name='bnhg1_low6_low6_low2_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low2_branch2c')
             .batch_normalization(name='bnhg1_low6_low6_low2_branch2c'))

        (self.feed('reshg1_low6_low6_low1_relu', 
                   'bnhg1_low6_low6_low2_branch2c')
             .add(name='reshg1_low6_low6_low2')
             .relu(name='reshg1_low6_low6_low2_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low5_branch2a')
             .batch_normalization(relu=True, name='bnhg1_low6_low6_low5_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low5_branch2b')
             .batch_normalization(relu=True, name='bnhg1_low6_low6_low5_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low5_branch2c')
             .batch_normalization(name='bnhg1_low6_low6_low5_branch2c'))

        (self.feed('reshg1_low6_low6_low2_relu', 
                   'bnhg1_low6_low6_low5_branch2c')
             .add(name='reshg1_low6_low6_low5')
             .relu(name='reshg1_low6_low6_low5_relu')
             .max_pool(2, 2, 2, 2, name='hg1_low6_low6_low6_pool1')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low6_low1_branch2a')
             .batch_normalization(relu=True, name='bnhg1_low6_low6_low6_low1_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low6_low1_branch2b')
             .batch_normalization(relu=True, name='bnhg1_low6_low6_low6_low1_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low6_low1_branch2c')
             .batch_normalization(name='bnhg1_low6_low6_low6_low1_branch2c'))

        (self.feed('hg1_low6_low6_low6_pool1', 
                   'bnhg1_low6_low6_low6_low1_branch2c')
             .add(name='reshg1_low6_low6_low6_low1')
             .relu(name='reshg1_low6_low6_low6_low1_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low6_low2_branch2a')
             .batch_normalization(relu=True, name='bnhg1_low6_low6_low6_low2_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low6_low2_branch2b')
             .batch_normalization(relu=True, name='bnhg1_low6_low6_low6_low2_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low6_low2_branch2c')
             .batch_normalization(name='bnhg1_low6_low6_low6_low2_branch2c'))

        (self.feed('reshg1_low6_low6_low6_low1_relu', 
                   'bnhg1_low6_low6_low6_low2_branch2c')
             .add(name='reshg1_low6_low6_low6_low2')
             .relu(name='reshg1_low6_low6_low6_low2_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low6_low5_branch2a')
             .batch_normalization(relu=True, name='bnhg1_low6_low6_low6_low5_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low6_low5_branch2b')
             .batch_normalization(relu=True, name='bnhg1_low6_low6_low6_low5_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low6_low5_branch2c')
             .batch_normalization(name='bnhg1_low6_low6_low6_low5_branch2c'))

        (self.feed('reshg1_low6_low6_low6_low2_relu', 
                   'bnhg1_low6_low6_low6_low5_branch2c')
             .add(name='reshg1_low6_low6_low6_low5')
             .relu(name='reshg1_low6_low6_low6_low5_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low6_low6_branch2a')
             .batch_normalization(relu=True, name='bnhg1_low6_low6_low6_low6_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low6_low6_branch2b')
             .batch_normalization(relu=True, name='bnhg1_low6_low6_low6_low6_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low6_low6_branch2c')
             .batch_normalization(name='bnhg1_low6_low6_low6_low6_branch2c'))

        (self.feed('reshg1_low6_low6_low6_low5_relu', 
                   'bnhg1_low6_low6_low6_low6_branch2c')
             .add(name='reshg1_low6_low6_low6_low6')
             .relu(name='reshg1_low6_low6_low6_low6_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low6_low7_branch2a')
             .batch_normalization(relu=True, name='bnhg1_low6_low6_low6_low7_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low6_low7_branch2b')
             .batch_normalization(relu=True, name='bnhg1_low6_low6_low6_low7_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low6_low7_branch2c')
             .batch_normalization(name='bnhg1_low6_low6_low6_low7_branch2c'))

        (self.feed('reshg1_low6_low6_low6_low6_relu', 
                   'bnhg1_low6_low6_low6_low7_branch2c')
             .add(name='reshg1_low6_low6_low6_low7')
             .relu(name='reshg1_low6_low6_low6_low7_relu')
             .deconv(4, 4, 256, 2, 2, biased=False, group=256, relu=False, name='hg1_low6_low6_low6_up5'))

        (self.feed('reshg1_low6_low6_low5_relu', 
                   'hg1_low6_low6_low6_up5')
             .add(name='hg1_low6_low6_low6')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low7_branch2a')
             .batch_normalization(relu=True, name='bnhg1_low6_low6_low7_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low7_branch2b')
             .batch_normalization(relu=True, name='bnhg1_low6_low6_low7_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='reshg1_low6_low6_low7_branch2c')
             .batch_normalization(name='bnhg1_low6_low6_low7_branch2c'))

        (self.feed('hg1_low6_low6_low6', 
                   'bnhg1_low6_low6_low7_branch2c')
             .add(name='reshg1_low6_low6_low7')
             .relu(name='reshg1_low6_low6_low7_relu')
             .deconv(4, 4, 256, 2, 2, biased=False, group=256, relu=False, name='hg1_low6_low6_up5'))

        (self.feed('reshg1_low6_low5_relu', 
                   'hg1_low6_low6_up5')
             .add(name='hg1_low6_low6')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low7_branch2a')
             .batch_normalization(relu=True, name='bnhg1_low6_low7_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='reshg1_low6_low7_branch2b')
             .batch_normalization(relu=True, name='bnhg1_low6_low7_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='reshg1_low6_low7_branch2c')
             .batch_normalization(name='bnhg1_low6_low7_branch2c'))

        (self.feed('hg1_low6_low6', 
                   'bnhg1_low6_low7_branch2c')
             .add(name='reshg1_low6_low7')
             .relu(name='reshg1_low6_low7_relu')
             .deconv(4, 4, 256, 2, 2, biased=False, group=256, relu=False, name='hg1_low6_up5'))

        (self.feed('reshg1_low5_relu', 
                   'hg1_low6_up5')
             .add(name='hg1_low6')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='reshg1_low7_branch2a')
             .batch_normalization(relu=True, name='bnhg1_low7_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='reshg1_low7_branch2b')
             .batch_normalization(relu=True, name='bnhg1_low7_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='reshg1_low7_branch2c')
             .batch_normalization(name='bnhg1_low7_branch2c'))

        (self.feed('hg1_low6', 
                   'bnhg1_low7_branch2c')
             .add(name='reshg1_low7')
             .relu(name='reshg1_low7_relu')
             .deconv(4, 4, 256, 2, 2, biased=False, group=256, relu=False, name='hg1_up5'))

        (self.feed('res6_relu', 
                   'hg1_up5')
             .add(name='hg1')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='linear1')
             .batch_normalization(relu=True, name='bn_linear1')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='output'))