# 参与 ContestTrade 贡献

首先，我们由衷地感谢您对 ContestTrade 项目的关注和贡献兴趣！

ContestTrade 是一个由社区驱动的开源项目，我们相信，正是社区的力量，才能赋予这个项目无限的生命力。每一份贡献，无论大小，都对我们至关重要。

这份指南将引导您如何参与到 ContestTrade 的开发中来。

**目录**
- [💬 交流与沟通](#-交流与沟通)
- [💡 如何贡献](#-如何贡献)
  - [🐛 报告 Bug](#-报告-bug)
  - [✨ 提出功能建议](#-提出功能建议)
  - [💻 贡献代码](#-贡献代码)
- [🛠️ 本地开发设置](#️-本地开发设置)
- [🚀 Pull Request 流程](#-pull-request-流程)


## 💬 交流与沟通

我们深知，有效的沟通是高效协作的基石。为了方便大家进行深入的技术交流、验证开发想法、协同解决问题，我们建立了专门的**核心开发者微信群**。

在您准备开始贡献代码或有任何技术疑问时，强烈建议您先加入社群，与我们进行沟通。我们每周会定期召开**线上开发者会议**，同步项目进展、讨论技术方案。

<p align="center">
  <img src="assets/contributor_group.jpg" style="width: 250px; height: auto;">
  <br>
  <small>备注“ContestTrade开发者”入群</small>
</p>

## 💡 如何贡献

我们欢迎任何形式的贡献。以下是一些参与社区的方式：

### 🐛 报告 Bug

如果您在使用中发现了 Bug，请通过 [GitHub Issues](https://github.com/FinStep-AI/ContestTrade/issues) 提交。为了让我们能更快地定位和解决问题，请在您的 Issue 中包含以下信息：

* **清晰的标题**，例如 `[Bug] 在XXX情况下，数据处理模块崩溃`。
* **详细的复现步骤**，描述您是如何触发这个 Bug 的。
* **期望的行为** 和 **实际发生的行为**。
* **您的运行环境信息**（操作系统、Python版本等）。
* 相关的**日志截图**或**错误堆栈信息**。

在提交前，请先搜索现有的 Issues，确保没有重复报告。

### ✨ 提出功能建议

如果您对 ContestTrade 有任何绝妙的想法或功能建议，也欢迎通过 [GitHub Issues](https://github.com/FinStep-AI/ContestTrade/issues) 与我们分享。请使用我们提供的 `Feature Request` 模板，并详细描述：

* **您希望增加什么功能？** 它解决了什么问题？
* **您的设想方案是什么？** 这个功能大概会如何工作？
* **是否有类似的参考案例？**

### 💻 贡献代码

如果您准备好直接贡献代码，我们感到万分激动！

* **寻找任务：** 您可以从 [Issues 列表](https://github.com/FinStep-AI/ContestTrade/issues) 中寻找您感兴趣的任务开始。
* **领取任务：** 在您决定开始一个任务前，请在对应的 Issue 下方留言，声明您将负责此任务，以避免多人重复工作。
* **沟通先行：** 对于比较大的Feature实现，请务必先通过 Issue 或在开发者群中与我们讨论方案，达成共识后再开始编码。

## 🛠️ 本地开发设置

1.  **Fork 仓库：** 点击项目主页右上角的 "Fork" 按钮，将主仓库 Fork 到您自己的账户下。
2.  **Clone 您的 Fork：**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/ContestTrade.git](https://github.com/YOUR_USERNAME/ContestTrade.git)
    cd ContestTrade
    ```
3.  **创建并激活环境 (推荐使用 Conda):**
    ```bash
    conda create -n contesttrade_dev python=3.10
    conda activate contesttrade_dev
    ```
4.  **安装依赖：**
    ```bash
    pip install -r requirements.txt
    # 建议同时安装开发依赖（如果未来有 requirements-dev.txt）
    ```
5.  **创建 `dev` 分支并切换：**
    ```bash
    git checkout -b dev origin/dev
    ```

## 🚀 Pull Request 流程

1.  **创建特性分支：** 从 `dev` 分支创建一个新的特性分支。请遵循清晰的命名规范，例如：
    * 新功能: `feat/add-us-stock-support`
    * 修复Bug: `fix/tushare-api-error`
    * 文档: `docs/update-contributing-guide`
    ```bash
    git checkout -b feat/your-feature-name
    ```
2.  **进行代码修改：** 在新的分支上进行开发和修改。
3.  **提交代码：** 使用清晰、规范的 Commit Message。我们推荐使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式。
    ```bash
    git add .
    git commit -m "feat: Add support for US stock market data"
    ```
4.  **推送至您的 Fork 仓库：**
    ```bash
    git push origin feat/your-feature-name
    ```
5.  **创建 Pull Request (PR)：**
    * 在 GitHub 上，前往您 Fork 的仓库，点击 "Contribute" -> "Open pull request"。
    * **请务必确保您的 PR 是提交到主仓库的 `dev` 分支，而不是 `main` 分支。**
    * 填写 PR 模板，清晰地描述您的改动，并关联相关的 Issue (例如 `Closes #123`)。
6.  **代码审查：** 我们会尽快审查您的 PR，并可能提出一些修改建议。请保持沟通，共同完善代码。
7.  **合并：** 审查通过后，我们会将您的代码合并到 `dev` 分支。恭喜您成为 ContestTrade 的贡献者！

再次感谢您的宝贵时间和贡献！