#include <iostream>

class Point {

    double x, y;
    char *label;

    public:
    //Constructor

    Point(double x_, double y_, string name):x(x_), y(y_) {
        label = new char[name.size() + 1];
        strcpy(label, name);
    }
    //Copy Constructor
    Point(const Point &other): x(other.x), y(other.y) {
        char *newLabel = other.label;
        label = newLabel;
    }

    //Assignment Operator
    Point& operator=(const Point &other) {
        if(this != other) {
            x = other.x;
            y = other.y;
            label = other.label;

        }
        return *this;
    }


    double mag() const {
        return (sqrt(x*x + y * y));
    }

    //In derived class
    
    //Move constructor

    //Move Assignment


    //Destructor

}


class specificPoint: public Point {

    double mag() override {
        return sqrt(x*x + y*y) + 1;
    }

    //Overloading
    double mag(double x1, double y1) {
        return sqrt(x1*x1 + y1*y1);
    }

}