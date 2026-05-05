#include <iostream>
#include <memory>
#include <string>

using namespace std;

class Point{

    int *data;
    double x, y;

    char *label;

    unique_ptr<myClass> m1;
    shared_ptr<myClass> m2;

    public:

    //Constructor
    Point(double x_, double y_, int *data_, 
        const string &name, unique_ptr<myClass> m1_, 
        shared_ptr<myClass> m2_):x(x_), y(y_), 
        m1(std::move(m1_)), m2(m2_) {
            //deep copy of data
            if(data_) {
                data = int new(*data_);
            }
            else {
                data = nullptr;
            }
            
            //deep copy of label
            label = new char[name.size() + 1];
            strcpy(label, name.c_str());            
    }

    //Copy constructor, data and label might be declared for other make a deep copy if present
    Point(const Point& other):x(other.x), y(other.y), m2(other.m2) {
        //data deep copy
        if(other.data) {
            data = new int(*other.data);
        }

        else {
            data = nullptr;
        }

        if(other.label) {
            label = new char[strlen(other.label) + 1];
            strcpy(label, other.label);
        }

        else {
            label = nullptr;
        }

        //deep copy for m1
        if(other.m1) {
            m1 = std::make_unique<myClass>(*other.m1);
        }

    }

    //Copy assignment operator, this.data, this.label might exist, free them first.
    Point& operator=(const Point& other) {
        if(this != other) {
            x = other.x;
            y = other.y;
            m2 = other.m2;
            delete data;
            data = other.data ? new int(*other.data) : nullptr;
            delete[] label;
            if(other.label) {
                label = new char[strlen(other.label) + 1];
                strcpy(label, other.label);
            }

            else {
                label = nullptr;
            }

            if(other.m1) {
                m1 = std::make_unique<myClass>(*other.m1);
            }

            else {
                m1.reset();
            }
        }

        return *this;
    }

    //Move constructor
    Point(const Point &&other) noexcept :
    x(other.x), y(other.y), m1(std::move(other.m1)),
    m2(std::move(other.m2)), label(other.label), data(other.data) {        
        other.data = nullptr;
        other.label = nullptr;
    }

    //Move assignment operator
    Point& operator=(const Point &&other) noexcept {
        if(this != other) {
            x = other.x;
            y = other.y;
            delete[] label;
            delete data;

            label = other.label;
            other.label = nullptr;

            data = other.data;
            other.data = nullptr;

            m1 = std::move(other.m1);
            m2 = std::move(other.m2);
            
        }
        
        
        return *this;

    }

    //Destructor

};