---
layout: post
title: C++文件遍历
category: 技术
tags: 编程语言
keywords: C++
description:
---

# 遍历指定文件夹下所有文件文件夹名，包括子文件夹

```C++
#include "stdafx.h"
#include <string>
#include <vector>
#include <io.h>
using namespace std;

void getAllFiles( string path, vector<string>& files) 
 { 
   //文件句柄 
   long  hFile  =  0; 
   //文件信息 
   struct _finddata_t fileinfo; 
   string p; 
   if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) != -1) 
   { 
     do 
     {
       if((fileinfo.attrib & _A_SUBDIR)) 
       { 
         if(strcmp(fileinfo.name,".") != 0 && strcmp(fileinfo.name,"..") != 0) 
         {
          files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
           getAllFiles( p.assign(path).append("\\").append(fileinfo.name), files ); 
         }
       } 
       else 
       { 
         files.push_back(p.assign(path).append("\\").append(fileinfo.name) ); 
       } 
     }while(_findnext(hFile, &fileinfo) == 0); 
     _findclose(hFile); 
   } 
 } 

int main(int argc, char* argv[])
{
	vector<string> files;
	getAllFiles(argv[1],files); //argv[1]参数为指定文件夹
}
```

# 读取指定文件夹下的当前文件夹名

```C++
#include "stdafx.h"
#include <string>
#include <vector>
#include <io.h>
using namespace std;

void getJustCurrentDir( string path, vector<string>& files) 
 { 
   //文件句柄 
   long  hFile  =  0; 
  //文件信息 
   struct _finddata_t fileinfo; 
   string p; 
   if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) != -1) 
   { 
     do 
     {
       if((fileinfo.attrib & _A_SUBDIR)) 
       { 
         if(strcmp(fileinfo.name,".") != 0 && strcmp(fileinfo.name,"..") != 0) 
         {
           files.push_back(fileinfo.name);
           //files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
         }
           
       }
     }while(_findnext(hFile, &fileinfo) == 0); 
     _findclose(hFile); 
   } 
 } 

int main(int argc, char* argv[])
{
	vector<string> files;
	getJustCurrentDir(argv[1],files);
}
```

# 读取指定文件夹下的当前文件名

```
#include "stdafx.h"
#include <string>
#include <vector>
#include <io.h>
using namespace std;

void getJustCurrentFile( string path, vector<string>& files) 
 { 
   //文件句柄 
   long  hFile  =  0; 
   //文件信息 
   struct _finddata_t fileinfo; 
   string p; 
   if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) != -1) 
   { 
     do 
     {
       if((fileinfo.attrib & _A_SUBDIR)) 
       { 
         ;
       } 
       else 
       { 
         files.push_back(fileinfo.name);
         //files.push_back(p.assign(path).append("\\").append(fileinfo.name) ); 
       }
     }while(_findnext(hFile, &fileinfo) == 0); 
     _findclose(hFile); 
   } 
 } 

int main(int argc, char* argv[])
{
	vector<string> files;
	getJustCurrentFile(argv[1],files);
}
```

# 只读取某给定路径下的所有文件名(即包含当前目录及子目录的文件)

```
#include "stdafx.h"
#include <string>
#include <vector>
#include <io.h>
using namespace std;

void getFilesAll( string path, vector<string>& files) 
 { 
   //文件句柄 
   long  hFile  =  0; 
   //文件信息 
   struct _finddata_t fileinfo; 
   string p; 
   if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) != -1) 
   { 
     do 
     {
       if((fileinfo.attrib & _A_SUBDIR)) 
       { 
         if(strcmp(fileinfo.name,".") != 0 && strcmp(fileinfo.name,"..") != 0) 
         {
           //files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
           getFilesAll( p.assign(path).append("\\").append(fileinfo.name), files ); 
         }
       } 
       else 
       { 
         files.push_back(p.assign(path).append("\\").append(fileinfo.name) ); 
       } 
     }while(_findnext(hFile, &fileinfo) == 0); 
     _findclose(hFile); 
   } 
 }

int main(int argc, char* argv[])
{
	vector<string> files;
	getFilesAll(argv[1],files);
}
```
