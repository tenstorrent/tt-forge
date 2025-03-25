# Contributing Guidelines for TT-Forge

Thank you for your interest in the TT-Forge project, we appreciate your support. This document covers how to contribute to TT-Forge. If you are new to TT-Forge, you may want to review our Getting Started Guide. 

If you are ready to make a contribution, follow this process:

1. Review the Contributions Checklist to ensure you've met all required standards.
2. Request appropriate write permissions by opening an issue <!--Add a list of templates-->.
3. Open a pull request. <!--Add info to this doc-->
4. If your submission requires documentation, include it. <!--Add info to this doc-->
5. Go through a code review. <!--Add info to this doc-->
6. Your contribution is merged into the repo. 

## Contributions Checklist 
Before making a contribution, make sure you've reviewed the standards in this checklist:

* As applicable:
    * [Coding Guidelines and Standards for C++](#coding-guidelines-and-standards-for-c)
    * [Coding Guidelines and Standards for Python](#coding-guidelines-and-standards-for-python)
* [File Structure and Format for Legal](#file-structure-and-format-for-legal)
* Testing 
    * Pre-Commit and Post-Commit
    * CI/CD Principles
* Debugging
* When to Include Documentation
* Git and Branch Naming Guidelines
* Pull Request Guidelines 

Each item in the checklist is explained in the subsequent sections. 

## Coding Guidelines and Standards for C++

This section outlines the coding standards used in the tt-forge project. These guidelines are designed to enhance the readability and maintainability of our shared codebase. While these guidelines are not strict rules for every situation, they are essential for maintaining consistency across the repository.

Our long-term aim is to have the entire codebase adhere to these conventions.

Since our compiler is built on the LLVM MLIR framework, we strive to align closely with the LLVM coding style guidelines outlined here: [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html).

In addition to these guidelines, we also provide suggestions for best practices when coding your contribution:

* [Best Practices for Contributions](/docs/contributing_best_practices.md)
* [Best Practices for Writing Error Messages](/docs/contributing_error_best_practices.md)

### Naming

Clear and descriptive names are crucial for code readability and preventing bugs. It’s important to choose names that accurately reflect the semantics and purpose of the underlying entities, within reason. Avoid abbreviations unless they are widely recognized. Once you settle on a name, ensure consistent capitalization throughout the codebase to avoid confusion.

The general naming rule is to use camel case for most names (e.g., WorkaroundPass, isRankedTensor())

* Type Names
    * Applies to classes, structs, enums, and typedefs.
    * Should be nouns that describe the entity's purpose.
    * Use upper camel case (e.g. TTNNOptimizerOptions, DecompositionPass).
* Variable Names
    * Should be nouns, as they represent state.
    * Use lower camel case (e.g. inputLayout).
* Function Names
    * Represent actions and should be verb phrases
    * Use lower camel case (e.g. createTTNNOptimizer(), emitTTNNAsCpp()).

### Includes

We prefer #includes to be listed in this order:

1. Main Module Header
2. Local/Private Headers
3. LLVM project/subproject headers (clang/..., lldb/..., llvm/..., etc)
4. System #includes

Each category should:
* Be sorted lexicographically by the full path.
* Be separated by a single blank line for clarity.

Only the [standard lib header](https://en.cppreference.com/w/cpp/header) includes should use <> whereas all the others should use quotes "". Additionally, all project headers must use absolute paths (rooted at ttmlir) to prevent preprocessor and namespacing issues. For example, prefer:

``` #include "ttmlir/module/something.h" ```

over:

``` #include "something.h" ```

Using TTIRToTTNN.cpp as an example, this is what includes would look like for us:

``` 
#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"  # main header

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"  # these are local/private headers
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"  # llvm project/subproj headers
#include "llvm/Support/LogicalResult.h"

#include <cstdio>  # system includes
#include <algorithm> 
```

### Comments

Write comments as full sentences, starting with a capital letter and ending with a period. Comments should explain why the code exists, not just what it does. Use comments to clarify logic, assumptions, or any non-obvious aspects of the code.

Example of a comment:

``` // Initialize the buffer to store incoming data from the network. ```

In general, C++ style comments (//) should be used. Use C-style comments (/**/) only for when documenting the significance of constants used as actual parameters in a call:

``` object.callFunction(/*arg0=*/nullptr); ```

Every function, class, or non-trivial piece of logic should have a comment. Avoid redundant comments for self-explanatory code, but never leave complex code unexplained. Example of redundant comment:

``` 
// Increment the counter by 1.  // Redundant, avoid. 
counter++; 
```

Ensure comments are accurate and reflect the current state of the code. Outdated or misleading comments can be worse than no comments at all.

### Code Denesting (Inversion)

Strive to minimize unnecessary indentation without compromising code clarity. One effective way to achieve this is by using early exits and the continue keyword in long loops.

Consider following example:

```
void doSomething(Operation *op)
{
    if (op->getNumOperands() > 0
        && isDpsOp(op)
        && doSomethingDifferent(op))
    {
        // ... some long code ...
    }
}
```

It is strongly recommended to format the code as follows:

```
void doSomething(Operation *op)
{
    // ...
    // We need to do something with the op that has more than 0 operands
    if (op->getNumOperands() <= 0 ) return;

    // We need something to do with the DPS op
    if (!isDpsOp(op)) return;

    // Just for example purposes
    if (!doSomethingDifferent(op)) return;

    // .. some long code ...
}
```

This reduces loop nesting, makes the reasoning behind the conditions clearer, and signals to the reader that there is no subsequent else to worry about, reducing cognitive load. This can significantly improve code readability and comprehension.

### Function Declaration and Definition Order

To improve code readability and maintainability, we should adopt a consistent approach for organizing function declarations and definitions within a file. The goal is to make it easier for readers to follow the logical flow of function dependencies.

Follow a bottom-up call order:
* Arrange functions so that lower-level helper functions are defined first, followed by higher-level functions that call them.
* This allows each function to be defined after its dependencies, making it clear which functions rely on which.
* For example, if function A calls A1 and A2, then the preferred order is:

```
void A1();
void A2();
void A(){
  A1();
  A2();
}
```

Group related functions together:
* If functions are only relevant to a specific “parent” function (e.g., A1 and A2 are only called by A), place them directly before the “parent” function.
* If a function (like A2) is also called by other functions (e.g., B), place it where it fits the overall bottom-up order.

Avoid mixed ordering:
* Mixing top-down and bottom-up call orders within the same file can make the code hard to read and maintain.

Example of a preferred order:

```
void A1() {
  /*...*/
}
void A2() {
  /*...*/
}
void B() {
  A2(); // A2 is defined before B, so dependencies are clear.
}
void A() {
  A1();
  A2();
  B();
}
```

### Helper Functions

This coding guideline addresses visibility and linkage of simple helper functions to ensure clarity, prevent linking errors, and improve maintainability:

If a helper function needs to be defined in a .cpp file, it should be declared static or wrapped inside an anonymous namespace. 

> [!NOTE]
> A significant concern with declaring functions as non-public (e.g., static functions 
> or functions in unnamed namespaces) is that they cannot be unit tested in isolation. 
> This limitation hinders our ability to write focused, granular tests that verify the 
> correctness of individual components and it also reduces test coverage.


If a helper function needs to be defined in a header file (e.g., for templated or performance-critical code), it should be marked as inline.

### Using Namespaces

Namespaces are an important part of C++ programming, providing a way to organize code and avoid naming conflicts. Choose namespace names that reflect the purpose or functionality of the code contained within.

Follow these guidelines when defining namespaces:

* Use lower-case letters for short, single-word names or those with a clear acronym (e.g., ttnn, mlir).
* Use nested namespaces to group logically related code, avoiding too deep or unnecessarily complex hierarchy

Follow these guidelines when using namespaces:

* Do not use a using-directive to make all names from a namespace available because it pollutes the namespace.

``` 
// Forbidden -- This pollutes the namespace.
using namespace std;
```

* Avoid placing code in the global namespace to reduce the potential for name conflicts and ambiguity. Always use specific namespaces. If necessary to use something from the global namespace (such as std), use an explicit std:: prefix rather than importing everything using using namespace std;.
* Do not use namespace aliases at namespace scope in header files except in explicitly marked internal-only namespaces, because anything imported into a namespace in a header file becomes part of the public API exported by that file.
* Try to avoid mixing concepts from different namespaces in a single function or class. If a function belongs to one namespace but calls classes from others, ensure the relationships are clear.
* Wrap classes/structs declared in .cpp files inside of an [anonymous namespace](https://en.cppreference.com/w/cpp/language/namespace#Unnamed_namespaces) to avoid violating [ODR](https://en.cppreference.com/w/cpp/language/definition). See [LLVM docs](https://llvm.org/docs/CodingStandards.html#anonymous-namespaces) for more detailed information.
Using Alternative Tokens (and, or, xor, etc.)

Although they are standard, we should avoid their use. They are very rarely used in practice and the C++ community widely uses the standard operators (&&, ||, !, etc.), as they are more familiar and easily recognizable to most C++ developers. Their usage can make the code harder to read and maintain, especially for developers who are not familiar with these alternatives. We should stick to the standard operators (&&, ||, !, etc.) for clarity, consistency, and compatibility with other C++ developers and tools.

### Type Aliasing

When declaring type aliases in C++ prefer using over typedef. using provides better readability, especially for complex types, and supports alias templates. Here is example:

```
// Preferred
using Callback = void(*)(int, double);

// Avoid
typedef void (*Callback)(int, double);
Choose alias names that clarify their role in the code. Avoid overly generic names that might obscure the type’s purpose, hence do not create a type alias unless it significantly improves clarity or simplifies complex types.
```

### Using Auto to Deduce Type

Use auto only when it enhances code readability or maintainability. Avoid defaulting to “always use auto.” Instead, apply it thoughtfully in the following scenarios: - When the type is immediately clear from the initializer, such as in cast<Foo>(...). - When the type is obvious from the context, making the code cleaner and more concise. - When the type is already abstracted, such as with container typedefs like std::vector<T>::iterator.

In all other cases, prefer explicit type declarations to maintain clarity and ensure the code remains easy to understand.

## Coding Guidelines and Standards for Python

### Python Version and Source Code Formatting
The current minimum version of Python required is 3.8 or higher. Python code in the LLVM repository should only use language features available in this version of Python.

The Python code within the LLVM repository should adhere to the formatting guidelines outlined in PEP 8.

For consistency and to limit churn, code should be automatically formatted with the black utility, which is PEP 8 compliant. Use its default rules. For example, avoid specifying --line-length even though it does not default to 80. The default rules can change between major versions of black. In order to avoid unnecessary churn in the formatting rules, we currently use black version 23.x in LLVM.

When contributing a patch unrelated to formatting, you should format only the Python code that the patch modifies. For this purpose, use the darker utility, which runs default black rules over only the modified Python code. Doing so should ensure the patch will pass the Python format checks in LLVM’s pre-commit CI, which also uses darker. When contributing a patch specifically for reformatting Python files, use black, which currently only supports formatting entire files.

Here are some quick examples, but see the black and darker documentation for details:

```
$ darker test.py                   # format uncommitted changes
$ darker -r HEAD^ test.py          # also format changes from last commit
$ black test.py                    # format entire file
```

Instead of individual file names, you can specify directories to darker, and it will find the changed files. However, if a directory is large, like a clone of the LLVM repository, darker can be painfully slow. In that case, you might wish to use git to list changed files. For example:

``` $ darker -r HEAD^ $(git diff --name-only --diff-filter=d HEAD^) ```

## File Structure and Format for Legal 

Every source file must have the appropriate Software Package Data Exchange (SPDX) header at the top. 

C++ header files follow the [Linux conventions](https://elixir.bootlin.com/linux/v6.5.1/source/Documentation/process/license-rules.rst#L71) for C++ source files, RST files, ASM files, and scripts. C++ header files should be treated as C++ source files and use this convention: 

```
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
```

Python files should use this convention: 

```
# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
```

<!----Not ready yet after this line
## STUFF ABOUT PULL REQUESTS 

For all your Pull Requests (PRs), Tenstorrent has an internal policy which your PR goes through after an initial review.

The initial review encompasses the following:
* Reviewing the PR for CI/CD readiness, making sure that the code and PR at a high level make sense for the project.
* Once approved for CI/CD readiness, a Tenstorrent developer kicks off the CI/CD pipeline on your behalf.

## Internal Contributions
For internal contributions Tenstorrent has the following guidelines:

* A 24 hour merge rule exists. Wait at least 24 hours after the PR was initially opened for review. This gives members of Tenstorrent teams that span the globe the opportunity to provide feedback on PRs.

In addition to the 24 hour rule, the following prerequisites for landing a PR exist:
* At least 1 reviewer signs off on the change
* Component owners must sign-off (GitHub will tell you if this hasn't been met)
* Green CI
* Wait at least 24 hours after opening the PR to give all tagged reviewers a chance to take a look, or at least comment on the issue that they need more time to review

```
> [!NOTE]
> *Rebasing or further changes to the PR do not reset the 24 hour counter.* 
``` --->

