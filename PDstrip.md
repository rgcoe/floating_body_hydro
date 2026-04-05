# PDstrip guide

- Download site: https://sourceforge.net/projects/pdstrip/
- GitHub clone: https://github.com/eriove/pdstrip

You can use the following steps to set up PDstrip.

1. Download `PDstrip` and compile if necessary (if you're using Windows, you will not need to compile because the `.exe` file comes with the package). Archives are available in `.zip` and `.gz` formats. Extract the files, taking note of where the `PDstrip` directory is created. Skim the manual provided as a `.ps` file in the subdirectory titled `doc` (and as a `.pdf` file that supplements the Topic 11 notes in Canvas), paying particular attention to Sections 1–4 and 20.

2. Open a command (or terminal) window and navigate to the `PDstrip` folder. Type `dir` (Windows users) or `ls` (Mac/Linux users) to list all files and subdirectories in the current directory. To “change directory,” type `cd`, then a space, then the name of the subdirectory you wish to move to. (Note: after typing the first letter or two of the subdirectory, pressing the tab key should autocomplete. If there are multiple subdirectories that start with the same letters, you may need to press the tab key multiple times to cycle through options.) The command `cd..` will move you up one level in the folder tree, i.e., closer to the root directory. The command `cd\` will take you to the root directory.

3. Once in the `PDstrip` directory, navigate to the `examples` folder. From the `examples` folder, use the command `..\pdstrip.exe pdstrip.inp` (Windows) or `../pdstrip.out pdstrip.inp` (Mac/Linux) to run `PDstrip` using the input file.

---

Based on guide originally developed by C. Woolsey and T. Battista
