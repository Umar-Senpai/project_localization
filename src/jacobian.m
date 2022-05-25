clear all
clc 
a = 2;
syms x y theta p_w phi_w;
%f = 5*x + phi_w*y^2;
f = phi_w - (cos((atan(y/x))-phi_w)*sqrt(y^2 + x^2));
f1 = diff(f,x);

g = phi_w - (cos((atan(y/x))-phi_w)*sqrt(y^2 + x^2));
g1 = diff(g,y)

w= cos((atan(y/x))-phi_w);
w1 = diff(w,y);