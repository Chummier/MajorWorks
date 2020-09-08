#include <iostream>
#include <string>
#include <vector>

using namespace std;

struct Node{
    int key;
    int parent;
    bool valid;
};

void prim(int** A, struct Node* G, int size, int root){
    for (int i = 0; i < size; i++){
        G[i].key = 99999999;
        G[i].parent = size;
        G[i].valid = true;
    }
    G[root].key = 0;
    int min;
    int u;
    int qSize = size;

    // while it hasn't visited every node yet
    while (qSize > 0){

        // u = ExtractMin(Q)
        min = 99999999;
        for (int i = 0; i < size; i++){
            if (G[i].key < min && G[i].valid){
                min = G[i].key;
                u = i;
            }
        }
        // Already visited u now
        G[u].valid = false;
        qSize--;

        // For all edges (u,v) out of u
        for (int i = 0; i < size; i++){
            if (A[u][i]){
                // If v hasn't been visited yet 
                if (G[i].valid && A[u][i] < G[i].key){
                    G[i].parent = u;

                    // DecreaseKey(Q, v, w(u,v))
                    G[i].key = A[u][i];
                }
            }
        }
    }
}

int main(){
    string input;
    int vertices, edges;

    getline(cin, input);

    vertices = stoi(input);

    getline(cin, input);
    edges = stoi(input);

    struct Node* graph = new struct Node[vertices];

    int** arr = new int*[vertices];
    for (int i = 0; i < vertices; i++){
        arr[i] = new int[vertices];
    }

    int a, b, c;

    for (int i = 0; i < edges; i++){
        getline(cin, input);
        a = stoi(input);
        b = stoi(input.substr(2));
        c = stoi(input.substr(4));

        arr[a][b] = c;
        arr[b][a] = c;
    }

    prim(arr, graph, vertices, 0);
    for (int i = 1; i < vertices; i++){
        cout << graph[i].parent << endl;
    }

    return 0;
}