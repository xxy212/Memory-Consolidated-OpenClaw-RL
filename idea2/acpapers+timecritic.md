<img width="2099" height="589" alt="image" src="https://github.com/user-attachments/assets/2cba6a57-b2ae-4a24-b216-47ed92fe2661" />
FSPO: FEW-SHOT OPTIMIZATION OF SYNTHETIC PREFERENCES PERSONALIZES TO REAL USERS  

Enhancing Personalized Multi-Turn Dialogue with Curiosity


A Personalized Agent with Adaptive Preference Arithmetic


Teaching Language Models to Evolve with Users: Dynamic Profile Modeling for Personalized Alignment

问题定义：当 personal agent 出错时，如何判断错误的具体原因？如自能力缺口、偏好误判还是记忆缺失。不同错误是否能够使用不同更新机制从而达到效率与性能的更好平衡？

设想结论：为了**让 personal agent 不是只会越来越会做事，而是越来越会“按这个人的方式做事” ，**个人Agent在优化过程中两方面需要持续有效进行改进：1  交互过程中完成用户布置任务所需的技能（metaClaw） 2 用户自身的偏好（包括回复偏好、决策偏好、路径偏好） 

核心方法：

1. 整体组件：
- 用户状态组件user state OR profile：记录用户长期、中期、短期的偏好；
- 双Skill库：Preference Skill:用户的做事原则 & Competence Skill:完成任务的经验
- Memrory:记录交互过程中事实性信息，以及交互过程中的摘要/总结
- 进化组件：错误归因组件、快进化组件、慢进化组件
- **错误归因组件**：任务：分析和用户交互的错误案例的原因，并根据不同类型的错误，选择不同优化方法；
- 快进化：直接对memory/skill/user state进行修改
- 慢进化：记录错误案例，结合修改后的正确memory/skill/user state进行 自举式Lora微调/ RL reward 有另一个模型决定


> 纪要：  去找怎么让agent完成任务更好的benchmark 这才是具体问题


