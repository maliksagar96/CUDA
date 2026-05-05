#include <iostream>
#include <memory>

using namespace std;

class Netflix {

    

    public:
        Netflix()        {
            cout << "Neflix Subscription taken for family.\n";
        }
        ~Netflix () { cout << "Netflix Subscription expired.\n" ;}

        void watch(const std::string &name) {cout<<name<<" is watching.\n";}

};


int main() {

    shared_ptr<Netflix> p1 = make_shared<Netflix>();
    p1->watch("p1");

    shared_ptr<Netflix> p2 = p1;
    p2->watch("p2");

    return 0;
}