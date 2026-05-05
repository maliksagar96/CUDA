#include <iostream>
#include <memory>

using namespace std;

class zoomCall {

    public:
        zoomCall() { 
            cout << "Zoom call started.\n";
        };

        ~zoomCall() {
            cout << "Zoom call ended.\n";
        }

        void admin() {
            cout << "I am admin.\n";
        }
};

int main() {

    std::unique_ptr<zoomCall> p1 = std::make_unique<zoomCall>();
    p1->admin();
    std::unique_ptr<zoomCall> p2 = std::move(p1);
    
    //Only p2 can be admin now.
    p2->admin();
    if(p1) {
        p1->admin();
    }
    

}