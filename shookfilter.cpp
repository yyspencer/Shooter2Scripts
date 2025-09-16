#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>

using namespace std;
namespace fs = filesystem;

// Function to check if a file has a .csv extension
bool isCSVFile(const fs::path& filePath) {
    return filePath.extension() == ".csv";
}

// Function to read a CSV file and check if it contains the target string
bool containsTargetString(const string& filePath, const string& target) {
    ifstream file(filePath);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open " << filePath << endl;
        return false; // File couldn't be opened
    }

    string line;
    while (getline(file, line)) { // Read each line
        stringstream ss(line);
        string cell;

        while (getline(ss, cell, ',')) { // Split by commas
            if (cell.find(target) != string::npos) { // Check if "shook" exists
                file.close(); return true;
            }
        }
    }
    file.close();
    return false;
}

int main() {
    string path = "."; // Default to current directory
    string targetString = "shook"; // String to search for
    fs::path shookFolder = fs::path(path) / "shook";

    // Ensure "shook" folder exists
    if (!fs::exists(shookFolder)) {
        fs::create_directory(shookFolder);
    }

    cout << "Scanning CSV files in: " << path << endl;

    for (const auto& entry : fs::directory_iterator(path)) {
        if (fs::is_regular_file(entry.path()) && isCSVFile(entry.path())) {
            string fileName = entry.path().filename().string();
            cout << "Checking file: " << fileName << endl;

            if (containsTargetString(entry.path().string(), targetString)) {
                // Move CSV file to the "shook" folder
                fs::path newFilePath = shookFolder / fileName;
                fs::rename(entry.path(), newFilePath);
                cout << "Moved " << fileName << " to " << shookFolder << endl;
            }
        }
    }

    cout << "\nProcessing complete." << endl;
    return 0;
}