#include <iostream>
#include <vector>
#include <string>

using namespace std;

vector<int> backup(6);
vector<int> frequency(6);
string try1;
string try2;

struct Node{
    int frequency;
    bool isDummy = false;
    struct Node* left = NULL;
    struct Node* right = NULL;
};

void revBubbleSort(){
    for (int i = 0; i < frequency.size(); i++){
        for (int j = 0; j < frequency.size()-1; j++){
            if (frequency[j+1] > frequency[j]){
                int temp = frequency[j+1];
                frequency[j+1] = frequency[j];
                frequency[j] = temp;
            }
        }
    }
}

void huffmanTree(struct Node* tree, int index){

    if (frequency.size() < 2){
        return;
    }

    // sort from highest to lowest so we can pop_back()

    revBubbleSort();

    // current two smallest tree roots added
    int temp = frequency[frequency.size()-1]+frequency[frequency.size()-2];

    // No current connections so join two smallest elements
    if (index == 6){
        tree[index].left = tree+5;
        tree[index].right = tree+4;
        tree[index].frequency = temp;
    } else {
        int node;
        for (int i = 0; i < index; i++){
            if (tree[i].frequency == frequency[frequency.size()-1]){
                node = i;
            }
        }
        tree[index].left = tree+node;

        for (int i = 0; i < index; i++){
            if (tree[i].frequency == frequency[frequency.size()-2]){
                if (i != node){
                    node = i;
                }

            }
        }
        tree[index].right = tree+node;
        tree[index].frequency = temp;
    }

    frequency.pop_back();
    frequency.pop_back();
    frequency.push_back(temp);

    huffmanTree(tree, ++index);
}

// way to print from https://www.geeksforgeeks.org/huffman-coding-greedy-algo-3/

void makeString(struct Node* tree, string s, int letter){
    if (tree == NULL){
        return;
    }

    if (tree->isDummy == false && tree->frequency == backup[letter]){
        string c;
        if (tree->frequency == backup[0]){ c = "A";} 
        if (tree->frequency == backup[1]){ c = "B";} 
        if (tree->frequency == backup[2]){ c = "C";} 
        if (tree->frequency == backup[3]){ c = "D";} 
        if (tree->frequency == backup[4]){ c = "E";} 
        if (tree->frequency == backup[5]){ c = "F";}
        
        cout << c << ":" << s << endl;
    }

    makeString(tree->left, s+"0", letter);
    makeString(tree->right, s+"1", letter);
}

int main(){
    struct Node* tree = new struct Node[11];

    for (int i = 0; i < 6; i++){
        cin >> frequency[i];
        backup[i] = frequency[i];
    }

    revBubbleSort();

    for (int i = 0; i < 6; i++){
        tree[i].frequency = frequency[i];
    }

    for (int i = 6; i < 11; i++){
        tree[i].isDummy = true;
    }
    
    huffmanTree(tree, 6);

    for (int i = 0; i < 6; i++){
        makeString(tree+10, "", i);
    }

    return 0;
}