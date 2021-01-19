//
// Created by cogito on 2021/1/14.
//
#include<bits/stdc++.h>
//#include<Python.h>
#ifndef ICRA_SHORTEST_PATH_SPFA_H
#define ICRA_SHORTEST_PATH_SPFA_H
#define N 510
#define M 810
using namespace std;

/*
class point {
    int x,y;
    point(int a,int b){
        x=a;y=b;
    }
    point operator+(const point &a){
        point box;
        box.x = this->x + a.x;
        box.y = this->y + a.y;
        return box;
    }
};
pair<int,int> operator+(const pair<int,int> &x, pair<int,int> &y){
    return make_pair(x.first+y.first,x.second+y.second);
}*/
class SPFA {
private:
    const int C=200;
    const double eps=1e-5;
    int n,m,d,x,y,xx,yy;
    char s[N][M];
    bool flag[N][M],ff[N][M];
    double f[N][M],value[N][M];
    pair<int,int> zz[N*M];
    int top_zz;
    int debug_modle=0;
    pair<int,int>seq[N*M*20],last[M][M],c[4],dd,z[N*M];
    void dfs(int x,int y){
        if (last[x][y].first!=0)
            dfs(last[x][y].first,last[x][y].second);
        z[++d]=make_pair(x,y);
    }
    void init() {
        int l = 0, r = 0;
        for (int i = 0; i <= n + 1; i++)
            for (int j = 0; j <= m + 1; j++) {
                value[i][j] = flag[i][j] = 0;
                if (s[i][j] != '0') {
                    flag[i][j] = 1;
                    seq[++r] = make_pair(i, j);
                }
            }
        while (l < r) {
            l++;
            for (int i = 0; i < 4; i++) {
                dd.first = seq[l].first + c[i].first;
                dd.second = seq[l].second + c[i].second;
                if (dd.first < 0 || dd.second < 0)continue;
                if (s[dd.first][dd.second] == '0' && !flag[dd.first][dd.second]) {
                    value[dd.first][dd.second] = value[seq[l].first][seq[l].second] + 1;
                    seq[++r] = dd;
                    flag[dd.first][dd.second] = 1;
                }
            }
        }
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)value[i][j] = max(1.0,200-value[i][j]*2);
    }
public:
    SPFA(int x,int y){
        memset(s,0,sizeof(s));
        memset(value,0,sizeof(value));
        memset(ff,0,sizeof(ff));
        n=x;m=y;d=0;top_zz=0;
        c[0].first=c[2].second=1;
        c[1].first=c[3].second=-1;
        c[0].second=c[2].first=c[1].second=c[3].first=0;
        for (int i=1;i<=n;i++)
            for (int j=1;j<=m;j++)
                s[i][j]='0';
    }
    void open_debug_modle(){
        debug_modle=1;//cout<<debug_modle<<endl;
        freopen("/home/cogito/CLionProjects/ICRA_shortest_path/debug_info.txt","w",stdout);
    }
    void set_begin_end(int X,int Y,int XX,int YY){
        x=X;y=Y;xx=XX;yy=YY;
        //cout<<x<<' '<<y<<' '<<xx<<' '<<yy<<endl;
    }
    void add_obstables(int X,int XX,int Y,int YY){
        //cout<<X<<' '<<XX<<' '<<Y<<' '<<YY<<endl;
        if (X>XX)swap(X,XX);if (Y>YY)swap(Y,YY);
        X-=30;Y-=30;XX+=30;YY+=30;
        X=min(X,0);XX=min(X,n);Y=max(Y,0);YY=min(YY,m);
        for (int i=X;i<=XX;i++)
            for (int j=Y;j<=YY;j++)s[i][j]='1';
    }
    int calc_SPFA(){
        init();
        int l=0,r=1;seq[r]=make_pair(x,y);
        for (int i=1;i<=n;i++)
            for (int j=1;j<=m;j++)f[i][j]=1e15;
        memset(flag,0,sizeof(flag));
        memset(ff,0,sizeof(ff));memset(last,0,sizeof(last));
        f[x][y]=0;
        while (l<r&&r<20*N*M-4){
            l++;for (int i=0;i<4;i++){
                dd.first = seq[l].first + c[i].first;
                dd.second = seq[l].second + c[i].second;
                if (s[dd.first][dd.second]=='0'&&f[dd.first][dd.second]+eps>
                                                 f[seq[l].first][seq[l].second]+value[seq[l].first][seq[l].second]){
                    f[dd.first][dd.second]=f[seq[l].first][seq[l].second]+value[seq[l].first][seq[l].second];
                    last[dd.first][dd.second]=seq[l];
                    ff[dd.first][dd.second]=1;
                    if (!flag[dd.first][dd.second]){
                        seq[++r]=dd;flag[dd.first][dd.second]=1;
                    }
                }
            }flag[seq[l].first][seq[l].second]=0;
        }
        if (ff[xx][yy]){
            d=0;dfs(xx,yy);

            if (debug_modle){
                cout<<d<<endl;
                for (int i=1;i<=d;i++){
                    //  write(z[i].first);putchar(' ');
                    //  writeln(z[i].second);
                    s[z[i].first][z[i].second]='2';
                }for (int i=1;i<=n;i++) {
                    for (int j = 1; j <= m; j++)putchar(s[i][j]);
                    puts("");
                }
            }return 1;
        }else{//=50
            for (int i=max(0,xx-50);i<=min(n,xx+50);i++)
                for (int j=max(0,yy-50);j<=min(m,yy+50);j++)
                    if (ff[i][j]){
                        xx=i;yy=j;
                        d=0;dfs(xx,yy);
                        return 1;
                    }
            puts("no_such_path");
            return 0;
            /*
            //freopen("/home/cogito/CLionProjects/ICRA_shortest_path/wrong_info.txt","w",stdout);
            if (debug_modle){
                for (int i=1;i<=n;i++) {
                    for (int j = 1; j <= m; j++)putchar(s[i][j]);
                    puts("");
                }
            }
            return 0;*/
        }
    }
    int smooth(int *a){
        int x=0,y=0,xx[N*M],yy[N*M],now_x=z[1].first,now_y=z[1].second,dd=0;
        //cout<<now_x<<' '<<now_y<<endl;
        for (int i=2;i<=d;i++){
            if (z[i].first-z[i-1].first){
                x+=z[i].first-z[i-1].first;
                if (y){
                    xx[++dd]=x;yy[dd]=y;x=y=0;
                    //cout<<xx[dd]<<' '<<yy[dd]<<endl;
                }
            }if (z[i].second-z[i-1].second){
                y+=z[i].second-z[i-1].second;
                if (x){
                    xx[++dd]=x;yy[dd]=y;x=y=0;
                    //cout<<xx[dd]<<' '<<yy[dd]<<endl;
                }
            }
        }if (x!=0 or y!=0){
            xx[++dd]=x;yy[dd]=y;x=y=0;
        }
        xx[dd+1]=yy[dd+1]=1e9;
        for (int i=1;i<=dd;){
            // cout<<now_x<<' '<<now_y<<endl;
            int j=i,dx=0,dy=0;
            for (;xx[j]==xx[i]&&yy[j]==yy[i];j++)dx+=xx[j],dy+=yy[j];
            now_x+=dx;now_y+=dy;
            i=j;zz[++top_zz]=make_pair(now_x,now_y);

            a[top_zz*2-1]=now_x;a[top_zz*2]=now_y;

            if (debug_modle)cout<<"new_route:"<<now_x<<' '<<now_y<<endl;
        }for (int i=1;i<=dd;i++)xx[i]+=xx[i-1],yy[i]+=yy[i-1];
        return top_zz;
    }
    float value_calc(int x,int y){
        return value[x][y];
    }
};
extern "C"{
// 重要，因为使用g++编译时函数名会改变，比方print_msg(const char*)
// 会编译成函数名 print_msg_char，这会导致python调用这个函数的时候
// 找不到对应的函数名，这有加了 extern "C"，才会以C语言的方式进行
// 编译，这样不会改变函数名字
SPFA* obj;
SPFA* create_SPFA(int x,int y){
    //cout<<x<<' '<<y<<endl;
    obj=new SPFA(x,y);
    return obj;
}
SPFA* recreate_SPFA(int x,int y){
    //cout<<x<<' '<<y<<endl;
    delete obj;
    obj=new SPFA(x,y);
    return obj;
}
void open_debug_modle(){
    obj->open_debug_modle();
}
void set_begin_end(int x,int y,int xx,int yy){
    obj->set_begin_end(x,y,xx,yy);
}
void add_obstables(int x,int xx,int y,int yy){
    obj->add_obstables(x,xx,y,yy);
}
int calc_SPFA(){
    return obj->calc_SPFA();
}
int smooth(int *a){
    return obj->smooth(a);
}
float value(int x,int y){
   // cout<<x<<' '<<y<<endl;
   // cout<<obj->value_calc(x,y)<<endl;
    return obj->value_calc(x,y);
}
}
#endif //ICRA_SHORTEST_PATH_SPFA_H