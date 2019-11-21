#include<iostream>
#include<vector>
#include<ctime>
#include<random>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <float.h>
using namespace std;
struct Parameters{
    int group_size;
    int dim;
    int M; //子群数量
    int iteration;///迭代次数
    double c1; //个体学习因子 
    double c2; //群体学习因子
    double k1; //逃逸标记调整阈值
    double k2; //阈值下降速度
    int ind; //函数选择
    Parameters(int a,int b, int c, int d, double e, double f, double g, double h, int j):group_size(a),dim(b),M(c),iteration(d),c1(e),c2(f),k1(g),k2(h), ind(j){}
};

Parameters para(20,30,5,6000,1.4,1.4,5,10,1); //from paper

const double pi = 3.14159;
const int P = para.group_size / para.M;
const double W = 200; //待优化变量的宽度

vector<vector<double> > group_v(para.group_size, vector<double>(para.dim)); //定义种群的个体速度
vector<vector<double> > group_x(para.group_size, vector<double>(para.dim)); //定义种群的个体位置
vector<vector<double> > pbest(para.group_size, vector<double> (para.dim)); //定义每个个体最优值
vector<double> gbest;//定义种群最优值
vector<double> P_fit; //子群适应度
vector<double> Td(para.dim, 0.5);//定义速度阈值 from paper
vector<double> Gd(para.dim);
vector<double> sd(para.M, W*2);//定义子群方差 


class SMO{
public: 
    void init(){
        for(int i=0; i<para.group_size; ++i){
            for(int j=0; j<para.dim; ++j){
                group_x[i][j] = (rand()*2.0/RAND_MAX-1)*W; //初始化每个粒子每个维度的位置
                pbest[i][j] = group_x[i][j]; //初始化每个粒子最优值
            }
        }
        sort();//升序
        gbest = pbest[0];
    }

    void sort(){
        vector<double> temp;
        for(int i=0; i<para.group_size; ++i){
            for(int j=i; j<para.group_size; ++j){
                if(fitness(group_x[i])>fitness(group_x[j])){
                    temp = group_x[i];
                    group_x[i] = group_x[j];
                    group_x[j] = temp;
                    
                    temp = pbest[i];
                    pbest[i] = pbest[j];
                    pbest[j] = temp;
                }
            }
        }
    }

    void update_sd(){
        double sum = 0;
        double fit_max = DBL_MIN;
        double fit_min = DBL_MAX;
        P_fit.clear();
        for(int i=0; i<para.M; ++i){
            double temp=0;
            for(int j=0; j<P;++j){
                temp += fitness(group_x[i*P+j]);
            }
            temp = temp / P*1.0;
            P_fit.push_back(temp);
            sum += temp;
            if(fit_max < temp) fit_max = temp;
            if(fit_min > temp) fit_min = temp;
        }
        //更新子群标准差
        for(int i=0; i<para.M; ++i){
            sd[i] *= exp((para.M*P_fit[i]-sum)/(fit_max-fit_min));
            //按8）式约束
            while(sd[i] > (W*1.0/4)){
                sd[i] = abs((W*1.0/4)-sd[i]);
            }
        }
    }
    //判断是否逃逸并更新
    void update_v(double w){
        double temp;
        double minf;
        double fit;
        double minx;
        double v_max;
        for(int i=0; i<para.group_size; ++i){
            for(int j=0; j<para.dim; ++j){
                if(group_v[i][j] == 0){ //HPSO
                    double r1=rand()*1.0/RAND_MAX;
                    int r2=1;
                    if(r1>0.5) r2=-1;
                    group_v[i][j] = r1*r2*pow(0.1,12);
                }
                //判断是否逃逸
                if(group_v[i][j]<Td[j]){
                    Gd[j]++;
                    temp = group_x[i][j];
                    minf = DBL_MAX;
                    for(int k=0; k<para.M; ++k){
                        group_x[i][j] = temp+gaussrand()*sd[k];
                        fit = fitness(group_x[i]);
                        if(minf>fit){
                            minf = fit;
                            minx = group_x[i][j];
                        }
                    }
                    v_max = W-abs(temp);
                    group_x[i][j] = temp+(rand()*2.0/RAND_MAX-1)*v_max;
                    if(minf<fitness(group_x[i])){
                        group_v[i][j] = minx-temp;
                    }
                    else{
                        group_v[i][j] = group_x[i][j]-temp;
                    }
                    group_x[i][j] = temp;
                }

                //更新速度和位置
                group_v[i][j] = w*group_v[i][j] + para.c1 * (rand()*1.0/RAND_MAX) * (pbest[i][j]-group_x[i][j]) + para.c2 * (rand()*1.0/RAND_MAX) * (gbest[j]-group_x[i][j]);
                group_x[i][j] += group_v[i][j];
                if(fitness(group_x[i]) < fitness(pbest[i])){
                    pbest[i]= group_x[i];
                }
                if(fitness(pbest[i]) < fitness(gbest)){
                    gbest = pbest[i];
                }
            }
        }
    }

    void update_td(){
        for(int i=0; i<para.dim; ++i){
            if(Gd[i]>para.k1){
                Gd[i] = 0;
                Td[i] /= para.k2;
            }
        }
    }
    
    double gaussrand()
    {
        double u = ((double)rand()/(RAND_MAX))*2-1;
        double v = ((double)rand()/(RAND_MAX))*2-1;
        double r=u*u+v*v;
        if(r==0 || r>1) return gaussrand();
        double c = sqrt(-2*log(r)/r);
        return u*c;
    }

    double fitness(vector<double>& temp){ //定义六个函数，根据int值来返回第几个函数的结果
        double result;
        switch (para.ind)
        {
        case 1:
            result = Tablet(temp);
            break;
        case 2:
            result = Quadric(temp);
            break;
        case 3:
            result = Rosenbrock(temp);
            break;
        case 4:
            result = Griewank(temp);
            break;
        case 5:
            result = Rastrigion(temp);
            break;
        case 6:
            result = Schaffer_F7(temp);
            break;
        }
        return result;
    }

    double Tablet(vector<double>& x){
        double res=pow(10,6) * x[0] * x[0];
        for(int i=1; i<para.dim; ++i){
            res += x[i] * x[i];
        }
        return res;
    }

    double Quadric(vector<double>& x){
        double res=0, value; 
        for(int i=0; i<para.dim; ++i){
            value = 0;
            for(int j=0; j<i; ++j){
                value += x[j];
            }
            res += value*value;
        }
        return res;
    }

    double Rosenbrock(vector<double>& a){
        double value = 0;
        for(int i=0;i<para.dim-1;i++){
            value += 100*(a[i+1]-a[i]*a[i])*(a[i+1]-a[i]*a[i])+(a[i]-1)*(a[i]-1);
        }
        return value;
    }

    double Griewank(vector<double>& a){
        double value1 = 0;
        double value2 = 1;
        for(int i=1;i<para.dim;i++){
            value1 += a[i]*a[i];
            value2  = value2 * cos(a[i]/sqrt(i));
        }
        return 1/4000 * value1 - value2 +1 ;

    }

    double Rastrigion(vector<double>& x){
        double res=1;
        for(int i=0; i<para.dim; ++i){
           res += x[i]*x[i]-10*cos(2*pi*x[i])+10;
        }
        return res;
    }

    double Schaffer_F7(vector<double>& a){
        double sum = 0;
        for(int i=0;i<para.dim-1;i++){
            sum+= pow(a[i]*a[i]+a[i+1]*a[i+1],0.25)*(sin(50*pow(a[i]*a[i]+a[i+1]*a[i+1],0.1))+1);
        }
        return sum;
    }


};

int main(){
    srand(time(NULL));
    SMO().init();
    double w;
    for(int i=0;i<para.iteration;++i){
        w = 0.95-0.45*i/para.iteration;
        //update sd
        SMO().update_sd();
        cout << "the "<<i+1<<" iteration's sd is ";
        for(auto &i : sd) cout<<i<<" ";
        cout << endl;
        //update v
        SMO().update_v(w);
        SMO().update_td();
        cout<<"the "<<i+1<<" iteration's best fitness is "<<SMO().fitness(gbest)<<endl;
    }
    cout<<"the best fitness is "<<SMO().fitness(gbest)<<endl;
    for(auto &i : gbest) cout<<i<<" ";
}
