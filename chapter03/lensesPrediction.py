# -*- coding: utf-8 -*-
# @Time    : 2018/12/17 17:28
# @Author  : GMell
# @Desc : ==============================================
# 通过决策树预测患者需要佩戴的隐形眼镜的类型
# Attribute Information:
# 数据集描述：
# -- result:class
# 1 : the patient should be fitted with hard contact lenses,
# 2 : the patient should be fitted with soft contact lenses,
# 3 : the patient should not be fitted with contact lenses.
# -- feature
# 1. age of the patient: (1) young, (2) pre-presbyopic(老花眼前期),
#                        (3) presbyopic(老花眼)
# 2. spectacle prescription: (1) myope(近视), (2) hypermetrope(远视)
# 3. astigmatic(散光): (1) no, (2) yes
# 4. tear production rate(眼泪比率): (1) reduced(稀少), (2) normal
# ======================================================
import numpy
import decisionTree
import treePlotter
def creatData(filename):
    fr = open(filename,'r+')
    lenses = [inst.strip().split('\t')  for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return lenses, lensesLabels

def main():
    lensesData, lensesLabels = creatData('lenseDataSet/lenses.txt')
    lensesTree = decisionTree.creatTree(lensesData, lensesLabels)
    print(lensesTree)
    treePlotter.createPlot(lensesTree)
if __name__ == '__main__':
    main()