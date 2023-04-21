---
layout: post
title: Windows下搭建github pages
category: 技术
tags: 技术
keywords: github pages
description: 环境搭建
---

# Windows下搭建github pages搭建

1. 在Windows上安装gith客户端——msysgit，网址http://msysgit.github.io/下载

2. 注册github帐号，然后在github上创建仓库，比如我的项目名字为manutdzou.github.io,这样以后访问网址manutdzou.github.io就能直接打开网页，github pages默认只能建立一个.io的项目能自动解析成博客。

3. 使用git clone https://github.com/xxxxxxx/xxxxx.git克隆到本地

4. 下载一个自己喜欢的模板放到自己的本地仓储中

5. 在本地编辑自己的项目

6. 打开msysgit（这是一个类似linux系统下的命令框，切换到本地仓储的目录里），git add . （将改动添加到暂存区）

7. git commit -m "提交说明"

8. git push origin master 将本地更改推送到远程master分支。

这样你就完成了向远程仓库的推送。如果在github的remote上已经有了文件，会出现错误。此时应当先pull一下，即：git pull origin master

然后再进行：git push origin master

这样跳出用户名和密码，依次输入就能完成推送。另外如果嫌每次都需要输入用户名和密码麻烦可以在本地生成私钥，在github账号处绑定就行。具体方法问度娘吧，比较简单。

# 下面介绍搭建Jekyll + GitHub Pages

## 安装 Ruby

首先，按需到 RubyInstallers 下载一个 Ruby 安装包，根据实际需求，选择“Ruby 2.3.0 (x64)”。

安装的时候注意勾选“Add Ruby executables to your PATH”，设置环境变量，这样一来，你将能在 Windows 命令行直接使用 Ruby 的相关命令。

![1](/public/img/posts/Github pages/1.png)

勾选 Add Ruby executables to your PATH

## 安装 Ruby DevKit

除此，由于 Jekyll 的一些依赖需要支持（例如 yajl-ruby），还需要安装一个 Ruby DevKit，Ruby 的开发工具包，一样在此 按需获取，选择 DevKit-mingw64-64-4.7.2-20130224-1432-sfx.exe。

这是一个压缩包， 为它建个目录（永久）并解压进去，例如 C:\RubyDevKit，进入此目录并初始化。

```
cd C:\RubyDevKit
ruby dk.rb init
```

若它不能自动获取 Ruby 目录时，需编辑其目录下的 config.yml 文件手动在后面加上

```
- C:/Ruby22-x64
```

最后安装 DevKit

```
ruby dk.rb install
```

## 安装 Jekyll

假如你打算将博客托管到 GitHub 上，建议直接跳到 github-pages,和 Linux 一样，在 Windows 上安装 Jekyll 仅需在命令行输入

```
gem install Jekyll
```

Ruby环境下的gem sources地址默认是国外网络地址，所以在使用gem的过程中经常会出现找不到资源的Error。那么如何解决这种Error？方法很简单, 要么就多次尝试执行gem命令，要么就修改gem源地址。

```
https:/rubygems.org/
http:/gems.github.com
http:/gems.rubyforge.com
http://ruby.taobao.org
gem sources –u 更新源
```

等待安装后，你就可以使用 Jekyll，使用 jekyll new 命令即可简单生成一个默认的博客，例如

```
jekyll new blog
```

## GitHub Pages Ruby Gem

GitHub 提供 github-pages 这个 gem，方便我们本地搭建和 GitHub Pages 线上相同的 Jekyll 环境，包括 Jekyll、少部分插件、Markdown 渲染引擎等等。

安装 gem

```
gem install github-pages
```

或许版本不够新，但一定最适合将博客托管在 GitHub Pages 的你。

## 语法高亮（可选）

若你的博客托管在 GitHub，又想使用语法高亮（pygments），那么你需要安装 Python。到 Python Releases for Windows 按需下载 Python 2，安装时和 Ruby 一样，如图注意勾选设置环境变量的选项。

![2](/public/img/posts/Github pages/2.png)

Python 设置环境变量

安装pip，下载git-pip.py放到python文件路径，运行git-pip.py安装pip,然后将python的Scripts路径添加到PATH.

## 安装 pygments

```
pip install pygments
```

安装对应的 gem

```
gem install pygments.rb
```

在配置中启用

```
highlighter: pygments
```

## 安装 wdm（可选）

从 v2.4.0 开始，Jekyll 本地部署时，会相当于以前版本加 --watch 一样，监听其源文件的变化，而 Windows 似乎有时候并不会奏效，不过鄙人使用并没碰到。当然你若碰到，可安装 wdm (Windows Directory Monitor ) 来改善这个问题。

这个分支 HaiderRazvi/wdm 可以被装上

git clone https://github.com/HaiderRazvi/wdm.git
cd wdm
gem build wdm.gemspec
gem install wdm-0.1.0.gem

说明一下，以上过程都是为了在本地配置github pages的环境，如果不是技术控，没必要那么麻烦，直接每次修改后push到github pages，网络上是自带全部配置环境的。

另外在写博客时候注意学习一下markdown的语法，公式编剧不建议用mathtype转格式，本人亲测好多问题。我使用mathjax，需要在网页配置时候在_layouts里的base.html里添加

```
<!--MathJax的配置脚本，用于临时简单的配置 -->
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    extensions: ["tex2jax.js"],
    <!--输入Latex公式，以HTML和CSS的形式显示输出 -->
    jax: ["input/TeX", "output/HTML-CSS"],
    tex2jax: {
      <!--$表示行内元素，$$表示块状元素 -->
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
      processEscapes: true
    },
    "HTML-CSS": { availableFonts: ["TeX"] }
  });
</script>
<!--加载MathJax的最新文件， async表示异步加载进来 -->
<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js">
</script>
```

另外推荐一个在线的latex公式编辑器 http://www.codecogs.com/eqnedit.php 我觉得非常好用。

另外记录一下当初开始选择写博客，主要觉得科研越做越顺手，工作也是研究性质的。可能需要花额外的时间去写博客，开始可能还会对环境，写作语法产生抵触。但是如果坚持下去，一点点积累总结自己做过的东西，这样也比较不容易忘记。

最后强烈推荐github pages,这里聚集了全世界的大牛，写作风格更加开放，学术氛围也比较好。比国内的很多博客的界面要友好，易于维护。

我们一直在github上索取，是时候贡献点东西了。


