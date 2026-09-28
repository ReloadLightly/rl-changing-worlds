# Reference implementation and attribution

`bandits.py` adapts epsilon-greedy and UCB action selection, optimistic
initialization, and incremental sample-average / constant-step-size learning ideas from
[Shangtong Zhang's `chapter02/ten_armed_testbed.py`](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter02/ten_armed_testbed.py),
consulted on 2026-09-28. The implementation here separates the learner from
the environment, vectorizes tasks, and adds shared environmental realizations,
nonstationarity, uncertainty estimates, and reproducible result recording.
UCB explicitly prioritizes untried actions instead of adding a small constant
to the count denominator. The constant-alpha UCB comparison is our variant;
it retains ordinary lifetime counts.
The source file's copyright and modification-permission declaration is
preserved at the top of `bandits.py`.

`gradient_bandits.py` adapts the reference's Section 2.8 softmax preference
update and current-reward-inclusive running baseline. It retains the same
authors' declaration. Our implementation uses stable softmax and vectorized
categorical sampling, and keeps preferences separate from action-value estimates.

`contextual_bandits.py` extends these attributed epsilon-greedy and gradient
updates with tables indexed by observed cue and cue-specific reward baselines.
It retains the authors' declaration. The two-context numerical experiment is
our original design inspired by Sutton and Barto, second edition, Section 2.9;
it is not a reproduction of a numerical experiment from the book.

The reference repository's [LICENSE](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/LICENSE)
is reproduced below for the adapted material:

MIT License

Copyright (c) 2019 Shangtong Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
