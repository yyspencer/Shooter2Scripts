#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;
using namespace std;

// Convert a string to lowercase
string toLower(const string& str) {
    string lowerStr = str;
    transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
    return lowerStr;
}

// Check if a file contains the target phrase
bool containsKeyword(const fs::path& filePath, const string& keyword) {
    ifstream file(filePath);
    if (!file.is_open()) return false;

    string line;
    while (getline(file, line)) {
        if (toLower(line).find(keyword) != string::npos) {
            return true;
        }
    }

    return false;
}

int main() {
    vector<string> folders = {"shook", "noshook"};
    string keyword = "robot entered survey room";
    fs::path surveyFolder = fs::current_path() / "survey";

    // Create survey folder if it doesn't exist
    if (!fs::exists(surveyFolder)) {
        fs::create_directory(surveyFolder);
    }

    int totalCSV = 0, foundCount = 0;
    vector<string> missingIndices;

    for (const auto& folder : folders) {
        fs::path folderPath = fs::current_path() / folder;
        if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) {
            cerr << "Warning: Folder '" << folder << "' does not exist or is not a directory.\n";
            continue;
        }

        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (fs::is_regular_file(entry) && entry.path().extension() == ".csv") {
                string fileName = entry.path().filename().string();
                if (fileName.length() < 5) continue;
                string index = fileName.substr(0, 5);
                totalCSV++;

                if (containsKeyword(entry.path(), keyword)) {
                    foundCount++;
                    // Move file to survey folder
                    fs::path newLocation = surveyFolder / fileName;
                    fs::rename(entry.path(), newLocation);
                } else {
                    missingIndices.push_back(index);
                }
            }
        }
    }

    // Output results
    cout << "\nTotal .csv files: " << totalCSV << "\n";
    cout << "Files containing keyword: " << foundCount << "\n";
    cout << "Files missing keyword (" << missingIndices.size() << "): ";
    for (const auto& idx : missingIndices) {
        cout << idx << " ";
    }
    cout << "\n";

    return 0;
}