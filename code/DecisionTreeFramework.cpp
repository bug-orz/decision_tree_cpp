#include"Read.h"
#include<iostream>
#include <iomanip>
#include<map>
#include<algorithm>
#include<set>
using namespace std;
//���ܺ�����ɾ��vector�����е�ĳһ��Ԫ��
vector<string> deleteElement(vector<string> data_array, string element) {
	for (vector<string>::iterator it = data_array.begin(); it != data_array.end(); )
	{
		if (*it == element)
		{
			it = data_array.erase(it); //����д��arr.erase(it);
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
    bool isLeaf = false;//�Ƿ���Ҷ�ӽڵ�
    int result = -1;//�����Ҷ�ӽڵ�Ļ�����Ӧ��label����
    vector<TreeNode*> branchs;//��֧�ڵ�
    int attr = -1;//����
    int attr_value = -1;//����ֵ   
};

class DecisionTree {
public:

    vector<vector<int>> trainData;//ѵ�����ݼ�����������
    vector<int> trainLabel;//ѵ�����ݼ���Ӧ�ı�ǩ
    map<int, set<int>> featureValues;//ÿ�����������
    TreeNode* decisionTreeRoot;//�������ĸ��ڵ�

    DecisionTree(vector<vector<int>>& trainData, vector<int>& trainLabel);
    void loadData(vector<vector<int>>& trainData, vector<int>& trainLabel);// ��������
    map<int, int> labelCount(vector<int>& dataset);//ͳ�����ݼ���ÿ����ǩ��������������Ϊ1�������ͽ��Ϊ2������
    double caculateEntropy(vector<int>& dataset);//������Ϣ��
    vector<int> splitDataset(vector<int>& dataset, int& feature, int& value);//�ָ����ݼ�
    double caculateGain(vector<int>& dataset, int& feature);//������Ϣ����
    int getMaxTimesLabel(map<int, int>& labelCount);//��ȡ���ִ������ı�ǩ
    int getMaxGainFeature(map<int, double>& gains);//��ȡ�����Ϣ���������
    TreeNode* createTree(vector<int>& dataset, vector<int>& features);//����������
    int classify(vector<int>& testData, TreeNode* root);
};

void DecisionTree::loadData(vector<vector<int>>& trainData, vector<int>& trainLabel) {
    //��������������������������ݼ���ǩ��������һ����ʱ������������
    if (trainData.size() != trainLabel.size()) {
        cerr << "input error" << endl;
        return;
    }
    //��ʼ��
    this->trainData = trainData;
    this->trainLabel = trainLabel;

    //����featureValues
    for (auto data : trainData) {
        for (int i = 0; i < data.size(); ++i) {
            featureValues[i].insert(data[i]);
        }
    }
}

map<int, int> DecisionTree::labelCount(vector<int>& dataset) {
    map<int, int> res;
    //�������ݼ���ͳ�Ʊ�ǩ���ֵĴ���
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
    //���������Ϊ�գ������Ϊ���ڵ��������Ϊ��ǩ�г��ִ������ı�ǩ
    if (features.size() == 0) {
        root->result = getMaxTimesLabel(label_count);
        root->isLeaf = true;
        return root;
    }
    //������ݼ���ֻ����һ�ֱ�ǩ�������Ϊ���ڵ��������Ϊ�ñ�ǩ
    if (label_count.size() == 1) {
        root->result = label_count.begin()->first;
        root->isLeaf = true;
        return root;
    }

    //������������ÿ����������Ϣ����
    map<int, double> gains;
    for (int feature : features) {
        gains[feature] = caculateGain(dataset, feature);
    }

    //��ȡ�����Ϣ�����������������Ϣ����
    int max_gain_feature = getMaxGainFeature(gains);
    vector<int> subFeatures = features;
    subFeatures.erase(find(subFeatures.begin(), subFeatures.end(), max_gain_feature));
    for (int value : featureValues[max_gain_feature]) {
        TreeNode* branch = new TreeNode();//������֧
        vector<int> subDataset = splitDataset(dataset, max_gain_feature, value);
        //����Ӽ�Ϊ�գ�����֧�ڵ���ΪҶ�ڵ㣬���Ϊ��ǩ�г��ִ������ı�ǩ
        if (subDataset.size() == 0) {
            branch->isLeaf = true;
            branch->result = getMaxTimesLabel(label_count);
            branch->attr = max_gain_feature;
            branch->attr_value = value;
            root->branchs.push_back(branch);
        }
        //����ݹ鴴����
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
    loadData(trainData, trainLabel);//��������
    vector<int> dataset(trainData.size());//���ݼ�
    for (int i = 0; i < trainData.size(); i++) {
        dataset[i] = i;
    }
    vector<int> features(trainData[0].size());//���Լ���
    for (int i = 0; i < trainData[0].size(); i++) {
        features[i] = i;
    }
    decisionTreeRoot = createTree(dataset, features);//����������
}

int DecisionTree::classify(vector<int>& testData, TreeNode* root) {
    //����������ڵ���Ҷ�ӽڵ㣬ֱ�ӷ��ؽ��
    if (root->isLeaf) {
        return root->result;
    }
    for (auto node : root->branchs) {
        //�ҵ���֧�����ڷ�֧����ϸ��
        if (testData[node->attr] == node->attr_value) {
            return classify(testData, node);
        }
    }
    return 0;
}


//////////////////////////////////////////////////////////
int main() {
	//���ݼ�������
	vector<string>  data_attributes;
    vector<string>  test_data_attributes;
	string train_file_name = "./restaurant_willwait_dataset/restaurant_willwait/restaurant_willwait_train.csv";
    string test_file_name = "./restaurant_willwait_dataset/restaurant_willwait/restaurant_willwait_test.csv";
	//��ȡѵ�����ݼ�����
	Read train_data = Read(train_file_name);
    Read test_data = Read(test_file_name);
	//��ȡ���ݼ��������б�
	data_attributes = train_data.dataHead;
    test_data_attributes = test_data.dataHead;
	//��ȡ���ݼ���ѵ������
	vector<vector<string>> train_data_table = train_data.dataSet;
    vector<vector<string>> test_data_table = test_data.dataSet;
	//�������ݼ�
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
    //��������
    // ���Լ�����ȷ���������
    //vector<double> test_result;
    //������ 1 �����ȴ� 0 �뿪
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
	//�������д���Ϊ����������������ʽ����ʱע���������д��롣
	//test_result.push_back(1);
	//classify_res.push_back(1);
	cout << "--------------------------result------------------------------" << endl;
	double res = evaluatScore(testLabel, classify_res);
	cout << "׼ȷ�ʣ�" << fixed << setprecision(2) << res << endl;
	cout << "--------------------------end------------------------------" << endl;

	system("pause");
	return 0;
}