# Contributing to Infinite-ISP
Infinite-ISP has an "open innovation" approach, which welcomes anyone with the right skills, time, and interest to contribute generously to executing the ISP framework from the software to the hardware level. It is a community-driven project so you can contribute whether it is:

- Adding algorithms or code to add features or improvements in the pipeline 

- Reporting a bug/issue
- Fix a bug
- Discussing the current state of the code
- Reviewing other developers' pull requests
- Improve documentation

## How to Contribute?
If you want to contribute to the project, we recommend you see the repo issues first and find the one that interests you. The pull request process is what we follow to merge any contributions, including bug fixes, new code or algorithm, new tutorials, document amendments, etc. To get started, the following steps must be taken.

1.	Install Git and set up your GitHub account. 

2.	Fork the infinite-isp repo and check [issues](https://github.com/xx-isp/infinite-isp/issues) to get started.
3.	Create a feature branch from the `main` branch to make changes after choosing a task.
4.	See the code guidelines to write your code. 
5.	Run the test cases to ensure the implemented code runs fine on your local system. 
6.	Make sure your code lints.
7.  Review the pull request process before sending the pull request.

## Bug Report or Feature Request

While contributing to our pipeline, we recommend investigating the repo [issues](https://github.com/xx-isp/infinite-isp/issues) first. For this, you are welcome to open an issue on finding any kind of bug or to request a feature inclusion. Just be sure it doesn’t comply with any existing problems or pull requests before sending an issue or new feature request. 

While reporting a bug, ensure it includes all the relevant information, like code snippets, estimators, functions, exceptions, etc., so anyone can reproduce the bug. 

If you want to add a new algorithm or request a new feature to the pipeline, you have to look for the following points:

1.	Techniques that will provide a clear-cut improvement in image quality, code efficiency, or optimization will be considered for inclusion.

2.	Algorithms should be computationally less expensive. 

3.	No compromise on quality benchmarks. 

## Pull Request Process
### PR Submission:

The pull request process for infinite-isp is purely based on merit. For making a good PR, one should read the following guidelines:
1.	After setting up the feature branch and adding your code to it, you must ensure it runs perfectly on your system without showing any bugs.

2.	Ensure you have followed the coding guidelines while implementing the algorithm.
3.	Your PR should represent one issue at a time. You should have created a distinct feature branch with a separate pull request for each bug or feature request.
4.	Squash all the unwanted commits before submitting the PR. 
5.	Add all the necessary information, like titles, meaningful variable names, comments explaining code lines, eloquent commit status, and relevant documentation.
6.	Add running tests and performance tests for the code before sending PR. 
7.	Avoid adding unnecessary datasets while sending PR. If you want to contribute quality datasets to the project, please generate a separate PR
8. Make sure the modified python files lints. We have implemented code checks for linting through Github Actions and Branch Protection Rules which means we cannot merge code into `main` if there are linting errors.

### PR Review and Merging

Two core developers are responsible for a PR review process. For a successful merger, it should be first approved by them. After carefully analyzing and considering the coding requirements, pertinent documentation, and quality assessments, they execute the validation and performance tests.

Then, if needed, you need to modify the code in response to the review you receive for your pull request. You must still change your branch by adding a new commit for this scenario. Push this commit to GitHub, and the pull request will be automatically updated.

## Coding Styles
We follow PEP 8 – style guide for python code as we develop this soft ISP in python.  The code should be well-commented, organized, and legible. 

### File Structure
Keep in mind the following norms while implementing the code.
- The modules folder should contain a separate python file for each new feature or functionality not currently in the pipeline. 

-   A new function should be introduced to the same class for any feature enhancement or updated algorithms for the same modules.
-   The helper functions should be put in the utils.py file. 
-   Follow the same coding pattern (PEP-8) in the modules files to prevent messy formatting or indentation. 
-   All the file names are written in lowercase for better compatibility. 
-   The algorithms implementations should be put in modules/<module name>, and their interfaces should be put in isp_pipeline.py. 
-	The relevant parameters should be updated in the config/<config.yml> file. 
-	Documentation is written in .pdf files, including all the references, images, and flowcharts, and should be put into infinite-isp/docs/.
-	Only English (ASCII) text is allowed for comment and explanation. No other language is allowed to use. 
### Naming convention 
-	All the file names should be written in lowercase.

-	All the class names should start with a capital letter, e.g., AutoWhiteBalance.
-	All the variable names should be in lowercase except those named after the author. 
-	An underscore separates words. 
-	Try using descriptive variables’ names for better understanding, e.g., start_index, upper_row, red_channel, etc. 
Code Commenting
Code comments are an important part of the code for getting an insight into the algorithms, so,
	Make sure all the comments are well-defined and comprehendible. 
	Complete sentences with a period in the end, should be used to clarify the meanings fully. The first letter should be in upper case. To explain any line of code, add an inline comment just before it. Example:


```python
      # Increment x by 1
      x = x + 1
```

-	Keep your code comments short and to the point. Use # for each line to make it easier to read, then format it as a paragraph. 

-	Docstrings should be added for public classes explaining what the method does, which is usually a one-liner docstring.
### Functions and Class Interfaces

-	The name of the functions and classes should be descriptive, explaining their purposes. 
-	All the functions should be well-defined with proper arguments and return values. 

## Code Review Guideline
Here is a checklist to go through while reviewing a code, 
1.  Evaluate the scope of the code and check its validity. 

2.	Check if the code is well written, i.e., following the coding guidelines.
3.	Make sure that the code is well-documented with an illustrative explanation.
4.	Running the test suite on the code to see if it fits precisely within our standards and quality benchmarks. 
5.	The code should be readable, clear, and non-redundant. 
6.	Check its dependencies and compatibility. 

## Code of Conduct

Since this initiative is community-driven, we expect positive behavior to lead to successful outcomes. We aim to provide a supportive, positive social atmosphere where community members may interact intellectually and exchange interests, skills, and valuable knowledge.
### Our Standards & Responsibilities
This is a civilized public forum where we want to uphold these norms:
1.	Be civil and respectful of the topics and the people discussing them, even if you disagree with some of what is being said.

2.	Criticize ideas, not people, so strictly avoid foul language.
3.	Provide reasoned counterarguments that improve the conversation.
4.	Help us influence the future of this community by choosing to engage in discussions that make this forum an exciting place to be — and avoiding those that do not.
5.	When you see bad behavior, don’t reply. It encourages bad conduct by acknowledging it, consumes your energy, and wastes everyone’s time. Just flag it. 
6.	This is a public forum, and search engines index these discussions. Keep the language, links, and images safe for family and friends.
7.	Make the effort to put things in the right place so that we can spend more time discussing and less cleaning up.
8.	You may not post anything digital that belongs to someone else without permission.
9.	Don’t post spam or otherwise vandalize the forum. Encourage the behavior you want to see in the world.
10.	We accept all who wish to participate in our activities, raising an environment where anyone can participate and make a difference.
## Enforcement
Project maintainers have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits, issues, and other contributions that are not aligned with this Code of Conduct or to ban temporarily or permanently any contributor for other behaviors that they deem inappropriate, threatening, offensive, or harmful.



## License
In short, when you submit code changes, your submissions are understood to be under the same [Apache 2.0 Lisence](LICENSE) that covers the project. Feel free to contact the maintainers if that's a concern.
License.

## References
1.	https://opensource.com/life/16/3/contributor-guidelines-template-and-tips
2.	https://pandas.pydata.org/docs/development/contributing_codebase.html#running-the-performance-test-suite
3.	https://scikit-learn.org/stable/developers/contributing.html 
4.	https://github.com/opencv/opencv/wiki/How_to_contribute 
5.	https://forum.opencv.org/faq/
6.	https://peps.python.org/pep-0008/#comments
