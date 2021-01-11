#include"Read.h"
#include<iostream>
#include <iomanip>
#include<map>
#include<algorithm>
#include<set>
using namespace std;
//功能函数：删除vector数组中的某一个元素
vector<string> deleteElement(vector<string> data_array, string element) {
	for (vector<string>::iterator it = data_array.begin(); it != data_array.end(); )
	{
		if (*it == element)
		{
			it = data_array.erase(it); //不能写成arr.erase(it);
		}
		else
		{
			++it;
		}
	}
	return data_array;
}

double evaluatScore(vector<double> test_result, vector<double> classify_res) {
	int correct_num = 0;
	for (int i = 0; i < test_result.size(); i++)
	{

		if (classify_res[i] == test_result[i]) {
			correct_num++;
		}
	}
	double sample_size = test_result.size();
	double res = correct_num / sample_size;
	return res;
}

//////////////////////////////////////////////////////////
struct TreeNode {
    bool isLeaf = false;//是否是叶子节点
    int result = -1;//如果是叶子节点的话，对应的label索引
    vector<TreeNode*> branchs;//分支节点
    int attr = -1;//特征
    int attr_value = -1;//特征值   
};

class DecisionTree {
public:

    vector<vector<int>> trainData;//训练数据集的特征数据
    vector<int> trainLabel;//训练数据集对应的标签
    map<int, set<int>> featureValues;//每个特征的类别
    TreeNode* decisionTreeRoot;//决策树的根节点

    DecisionTree(vector<vector<int>>& trainData, vector<int>& trainLabel);
    void loadData(vector<vector<int>>& trainData, vector<int>& trainLabel);// 导入数据
    map<int, int> labelCount(vector<int>& dataset);//统计数据集中每个标签的数量，比如结果为1的数量和结果为2的数量
    double caculateEntropy(vector<int>& dataset);//计算信息熵
    vector<int> splitDataset(vector<int>& dataset, int& feature, int& value);//分割数据集
    double caculateGain(vector<int>& dataset, int& feature);//计算信息增益
    int getMaxTimesLabel(map<int, int>& labelCount);//获取出现次数最多的标签
    int getMaxGainFeature(map<int, double>& gains);//获取最大信息增益的特征
    TreeNode* createTree(vector<int>& dataset, vector<int>& features);//创建决策树
    int classify(vector<int>& testData, TreeNode* root);
};

void DecisionTree::loadData(vector<vector<int>>& trainData, vector<int>& trainLabel) {
    //如果数据特征向量的数量和数据集标签的数量不一样的时候，数据有问题
    if (trainData.size() != trainLabel.size()) {
        cerr << "input error" << endl;
        return;
    }
    //初始化
    this->trainData = trainData;
    this->trainLabel = trainLabel;

    //计算featureValues
    for (auto data : trainData) {
        for (int i = 0; i < data.size(); ++i) {
            featureValues[i].insert(data[i]);
        }
    }
}

map<int, int> DecisionTree::labelCount(vector<int>& dataset) {
    map<int, int> res;
    //遍历数据集，统计标签出现的次数
    for (int index : dataset) {
        res[trainLabel[index]]++;
    }
    return res;
}

double DecisionTree::caculateEntropy(vector<int>& dataset) {
    map<int, int> label_count = labelCount(dataset);
    int len = dataset.size();
    double result = 0;
    for (auto count : label_count) {
        double pi = count.second / static_cast<double>(len);
        result -= pi * log2(pi);
    }
    return result;
}

vector<int> DecisionTree::splitDataset(vector<int>& dataset, int& feature, int& value) {
    vector<int> res;
    for (int index : dataset) {
        if (trainData[index][feature] == value) {
            res.push_back(index);
        }
    }
    return res;
}

double DecisionTree::caculateGain(vector<int>& dataset, int& feature) {
    set<int> values = featureValues[feature];
    double result = 0;
    for (int value : values) {
        vector<int> subDataset = splitDataset(dataset, feature, value);
        result += subDataset.size() / static_cast<double>(dataset.size()) * caculateEntropy(subDataset);
    }
    return caculateEntropy(dataset) - result;

}

int DecisionTree::getMaxTimesLabel(map<int, int>& labelCount) {
    int max_count = 0;
    int res;
    for (auto label : labelCount) {
        if (max_count <= label.second) {
            max_count = label.second;
            res = label.first;
        }
    }
    return res;
}

int DecisionTree::getMaxGainFeature(map<int, double>& gains) {
    double max_gain = 0;
    int max_gain_feature;
    for (auto gain : gains) {
        if (max_gain <= gain.second) {
            max_gain = gain.second;
            max_gain_feature = gain.first;
        }
    }
    return max_gain_feature;
}

TreeNode* DecisionTree::createTree(vector<int>& dataset, vector<int>& features) {
    TreeNode* root = new TreeNode();
    map<int, int> label_count = labelCount(dataset);
    //如果特征集为空，则该树为单节点树，类别为标签中出现次数最多的标签
    if (features.size() == 0) {
        root->result = getMaxTimesLabel(label_count);
        root->isLeaf = true;
        return root;
    }
    //如果数据集中只包含一种标签，则该树为单节点树，类别为该标签
    if (label_count.size() == 1) {
        root->result = label_count.begin()->first;
        root->isLeaf = true;
        return root;
    }

    //计算特征集中每个特征的信息增益
    map<int, double> gains;
    for (int feature : features) {
        gains[feature] = caculateGain(dataset, feature);
    }

    //获取最大信息增益的特征和最大的信息增益
    int max_gain_feature = getMaxGainFeature(gains);
    vector<int> subFeatures = features;
    subFeatures.erase(find(subFeatures.begin(), subFeatures.end(), max_gain_feature));
    for (int value : featureValues[max_gain_feature]) {
        TreeNode* branch = new TreeNode();//创建分支
        vector<int> subDataset = splitDataset(dataset, max_gain_feature, value);
        //如果子集为空，将分支节点标记为叶节点，类别为标签中出现次数最多的标签
        if (subDataset.size() == 0) {
            branch->isLeaf = true;
            branch->result = getMaxTimesLabel(label_count);
            branch->attr = max_gain_feature;
            branch->attr_value = value;
            root->branchs.push_back(branch);
        }
        //否则递归创建树
        else {
            branch = createTree(subDataset, subFeatures);
            branch->attr = max_gain_feature;
            branch->attr_value = value;
            root->branchs.push_back(branch);
        }
    }
    return root;
}

DecisionTree::DecisionTree(vector<vector<int>>& trainData, vector<int>& trainLabel) {
    loadData(trainData, trainLabel);//导入数据
    vector<int> dataset(trainData.size());//数据集
    for (int i = 0; i < trainData.size(); i++) {
        dataset[i] = i;
    }
    vector<int> features(trainData[0].size());//属性集合
    for (int i = 0; i < trainData[0].size(); i++) {
        features[i] = i;
    }
    decisionTreeRoot = createTree(dataset, features);//创建决策树
}

int DecisionTree::classify(vector<int>& testData, TreeNode* root) {
    //如果决策树节点是叶子节点，直接返回结果
    if (root->isLeaf) {
        return root->result;
    }
    for (auto node : root->branchs) {
        //找到分支，并在分支中再细分
        if (testData[node->attr] == node->attr_value) {
            return classify(testData, node);
        }
    }
    return 0;
}


//////////////////////////////////////////////////////////
int main() {
	//数据集的属性
	vector<string>  data_attributes;
    vector<string>  test_data_attributes;
	string train_file_name = "./restaurant_willwait_dataset/restaurant_willwait/restaurant_willwait_train.csv";
    string test_file_name = "./restaurant_willwait_dataset/restaurant_willwait/restaurant_willwait_test.csv";
	//获取训练数据集对象
	Read train_data = Read(train_file_name);
    Read test_data = Read(test_file_name);
	//获取数据集的属性列表
	data_attributes = train_data.dataHead;
    test_data_attributes = test_data.dataHead;
	//获取数据集的训练数据
	vector<vector<string>> train_data_table = train_data.dataSet;
    vector<vector<string>> test_data_table = test_data.dataSet;
	//测试数据集
	//vector<vector<string>> test_samples; 
    ///////////////////////////////////////
    cout << "start:   " << endl;
    vector<vector<int>> trainData;
    vector<int> trainLabel;
    for (int i = 0; i < train_data_table.size(); i++) {
        vector<int> temp;
        for (int j = 0; j < train_data_table[0].size() - 1; j++) {
            temp.push_back(atof(train_data_table[i][j].c_str())*1000);
        }
        trainLabel.push_back(atof(train_data_table[i][train_data_table[0].size() - 1].c_str()));
        trainData.push_back(temp);
    }
    DecisionTree dt = DecisionTree(trainData, trainLabel);
    //测试数据
    // 测试集合正确的类别数组
    //vector<double> test_result;
    //分类结果 1 继续等待 0 离开
    vector<double> classify_res;
    vector<double> testLabel;
    TreeNode* root = dt.decisionTreeRoot;
    for (int i = 0; i < test_data_table.size(); i++) {
        vector<int> temp;
        for (int j = 0; j < test_data_table[0].size() - 1; j++) {
            temp.push_back(atof(test_data_table[i][j].c_str())*1000);
        }
        int type = dt.classify(temp, root);
        cout << type << "    ";
        classify_res.push_back(type);
        testLabel.push_back(atof(test_data_table[i][test_data_table[0].size() - 1].c_str()));
        cout<< atof(test_data_table[i][test_data_table[0].size() - 1].c_str()) <<endl;
    }
	//以下两行代码为测试用例，请在正式调用时注释以下两行代码。
	//test_result.push_back(1);
	//classify_res.push_back(1);
	cout << "--------------------------result------------------------------" << endl;
	double res = evaluatScore(testLabel, classify_res);
	cout << "准确率：" << fixed << setprecision(2) << res << endl;
	cout << "--------------------------end------------------------------" << endl;

	system("pause");
	return 0;
}