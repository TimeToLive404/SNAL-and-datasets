import torch.nn as nn


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


# 读取模型的配置文件，返回网络超参数和网络结构
def model_load(model_cfg):
    # 读取出模型的配置文件的内容
    with open(model_cfg, 'r') as f:
        lines = f.read().split('\n')  # 分行，顺便去掉\n，从这个角度说，比使用readlines好
        lines = [x for x in lines if len(x) > 0]  # 去掉空行
        lines = [x for x in lines if x[0] != '#']  # 去掉注释行
    # 将每个块放入列表保存
    block = {}  # 将一个块的有效的内容保存为字典
    blocks = []  # 将所有块保存为一个列表
    for line in lines:

        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1]
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    module_list = nn.ModuleList()
    prev_filters = 3
    for i, block in enumerate(blocks[1:]):
        module = nn.Sequential()
        if block['type'] == 'convolutional':
            try:
                batch_normalize = block['batch_normalize']
                bais = False
            except:
                batch_normalize = 0
                bais = True
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            if block['pad']:  # 检查是否需要pad
                padding = (int(block['size']) - 1) // 2
            else:
                padding = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=bais)
            module.add_module(f'Conv_{i}', conv)
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f'bn_{i}', bn)
            if block['activation'] == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f'LeakyReLU_{i}', activn)
        elif block['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module(f'shortcut_{i}', shortcut)
        module_list.append(module)
        prev_filters = filters

    return blocks, module_list


class MyCnn(nn.Module):
    def __init__(self, module_cfg='./config/c53.cfg'):
        super(MyCnn, self).__init__()
        # 来自darknet的结构
        self.blocks, self.module_list = model_load(module_cfg)
        self.net_cfg = self.blocks[0]
        # 自定义的结构
        self.classifier = nn.Sequential(
            nn.Linear(173056, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x, get_feature=0):  # , visual_fmap=None
        outputs = {}
        for i, module in enumerate(self.blocks[1:]):
            if module['type'] == 'convolutional':  # 进行卷积计算
                x = self.module_list[i](x)
                if i == 73:
                    last_conv = x


            elif module['type'] == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i - 1] + outputs[i + from_]  # 残差连接结构

            outputs[i] = x
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        get_feature += 1
        return last_conv,output

    def weight_load(self, weight):
        ...


def model_size(x):
    ...


# 测试代码
net = MyCnn()
print(net)
