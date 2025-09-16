#include <iostream>
#include <filesystem>

using namespace std;
namespace fs = filesystem;

// Function to check if a file is valid based on its filename
bool isValidFile(const string& fileName) {
    return (fileName.find("Standard_School") != string::npos) && 
           (fileName.find("Tablet") == string::npos);
}

// Function to check if a file has a .csv extension
bool isCSVFile(const fs::path& filePath) {
    return filePath.extension() == ".csv";
}

int main() {
    string path = "."; // Default to current directory
    fs::path noshookFolder = fs::path(path) / "noshook";

    cout << "Scanning CSV files in the current directory..." << endl;

    // Check if the noshook folder exists; if not, create it
    if (!fs::exists(noshookFolder)) {
        cout << "Creating folder: 'noshook'..." << endl;
        fs::create_directory(noshookFolder);
    }

    for (const auto& entry : fs::directory_iterator(path)) {
        if (fs::is_regular_file(entry.path()) && isCSVFile(entry.path())) {
            string fileName = entry.path().filename().string();

            // Check if the file is valid
            if (isValidFile(fileName)) {
                fs::path destination = noshookFolder / fileName;

                // Check if the file already exists in noshook
                if (fs::exists(destination)) {
                    cout << "Skipping: " << fileName << " (Already exists in 'noshook')" << endl;
                } else {
                    // Move the file
                    fs::rename(entry.path(), destination);
                    cout << "Moved: " << fileName << " -> 'noshook' folder" << endl;
                }
            }
        }
    }

    cout << "\nProcessing complete." << endl;
    return 0;
}