#include <cstdlib>
#include <iostream>

#define VALUE(string) #string
#define TO_LITERAL(string) VALUE(string)

int main(int argc, char** argv) {

    std::string python = "python3";
    std::string script = TO_LITERAL(PROJECT_SOURCE_DIR) + std::string("src/nerf/main.py");  // Adjust if needed

    std::string cmd = python + " -m pip install -r ../requirements.txt && " + python + " " + script;

    std::cout << "Launching Python pipeline...\n";
    int result = std::system(cmd.c_str());
    return result;
}
