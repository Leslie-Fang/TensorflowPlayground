#include "iostream"

using namespace std;

extern "C" int add(int x,int y){
    cout<<"prepare to add!"<<endl;
    return x+y;
}

int main(){

    cout<<"hello world"<<endl;
    return 0;
}