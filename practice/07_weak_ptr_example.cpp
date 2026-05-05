#include <iostream>
#include <memory>
#include <string>

using namespace std;

class Netflix {

public:
  Netflix() {
    cout << "Netflix Subscription taken for family.\n";
  }

  ~Netflix() {
    cout << "Netflix Subscription expired.\n";
  }

  void watch(const string& name) {
    cout << name << " is watching.\n";
  }
};

int main() {

  shared_ptr<Netflix> p1 = make_shared<Netflix>();
  shared_ptr<Netflix> p2 = p1;

  weak_ptr<Netflix> guest = p1;   // does NOT own subscription

  p1->watch("p1");
  p2->watch("p2");

  cout << "Owners: " << p1.use_count() << "\n"; // 2

  // p1 unsubscribes
  p1.reset();
  cout << "p1 unsubscribed\n";

  // guest checks if subscription still exists
  if (shared_ptr<Netflix> temp = guest.lock()) {
    temp->watch("guest");
  } else {
    cout << "No active subscription\n";
  }

  // p2 unsubscribes (last owner)
  p2.reset();

  if (shared_ptr<Netflix> temp = guest.lock()) {
    temp->watch("guest");
  } else {
    cout << "No active subscription\n";
  }

  // guest tries again
  if (guest.expired())
    cout << "Subscription fully expired\n";
}