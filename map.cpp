#include <iostream>
#include <filesystem>
#include <set>
#include <iomanip> // For setting decimal precision

using namespace std;
namespace fs = filesystem;

// Function to check if a file has a .csv extension
bool isCSVFile(const fs::path& filePath) {
    return filePath.extension() == ".csv";
}

// Function to extract the first 5 characters as an index
string extractIndex(const string& fileName) {
    return fileName.substr(0, 5); // First 5 characters represent the index
}

int main() {
    string path = "."; // Default to current directory
    fs::path shookFolder = fs::path(path) / "shook";
    fs::path noshookFolder = fs::path(path) / "noshook";

    // Sets to store unique indices
    set<string> uniqueIndicesCurrent; // Unique indices in current directory
    set<string> uniqueIndicesShook;   // Unique indices in shook folder
    set<string> uniqueIndicesNoshook; // Unique indices in noshook folder
    set<string> missingIndices;       // Indices missing from both shook and noshook

    cout << "Scanning CSV files in the current directory..." << endl;

    // Scan the current directory for unique indices
    for (const auto& entry : fs::directory_iterator(path)) {
        if (fs::is_regular_file(entry.path()) && isCSVFile(entry.path())) {
            string fileName = entry.path().filename().string();
            string fileIndex = extractIndex(fileName);
            uniqueIndicesCurrent.insert(fileIndex);
        }
    }

    cout << "Scanning CSV files in the shook folder..." << endl;

    // Scan the shook folder for corresponding indices
    if (fs::exists(shookFolder) && fs::is_directory(shookFolder)) {
        for (const auto& entry : fs::directory_iterator(shookFolder)) {
            if (fs::is_regular_file(entry.path()) && isCSVFile(entry.path())) {
                string fileName = entry.path().filename().string();
                string fileIndex = extractIndex(fileName);
                if (uniqueIndicesCurrent.count(fileIndex)) {  // Only track if it was in the current directory
                    uniqueIndicesShook.insert(fileIndex);
                }
            }
        }
    }

    cout << "Scanning CSV files in the noshook folder..." << endl;

    // Scan the noshook folder for corresponding indices
    if (fs::exists(noshookFolder) && fs::is_directory(noshookFolder)) {
        for (const auto& entry : fs::directory_iterator(noshookFolder)) {
            if (fs::is_regular_file(entry.path()) && isCSVFile(entry.path())) {
                string fileName = entry.path().filename().string();
                string fileIndex = extractIndex(fileName);
                if (uniqueIndicesCurrent.count(fileIndex)) {  // Only track if it was in the current directory
                    uniqueIndicesNoshook.insert(fileIndex);
                }
            }
        }
    }

    cout << "\nChecking data completeness...\n";

    // Identify missing indices (found in the current directory but NOT in shook or noshook)
    for (const string& index : uniqueIndicesCurrent) {
        if (uniqueIndicesShook.count(index) == 0 && uniqueIndicesNoshook.count(index) == 0) {
            missingIndices.insert(index);
        }
    }

    // Print results
    cout << "\n==== Data Categorization Report ====\n";

    // List indices in shook
    cout << "\nIndices with shook files (" << uniqueIndicesShook.size() << "):\n";
    for (const string& index : uniqueIndicesShook) {
        cout << index << " ";
    }
    cout << "\n";

    // List indices in noshook
    cout << "\nIndices with noshook files (" << uniqueIndicesNoshook.size() << "):\n";
    for (const string& index : uniqueIndicesNoshook) {
        cout << index << " ";
    }
    cout << "\n";

    // List missing indices (not in either folder)
    cout << "\nIndices missing from both shook and noshook (" << missingIndices.size() << "):\n";
    for (const string& index : missingIndices) {
        cout << index << " ";
    }
    cout << "\n";

    // Calculate percentages
    int totalIndices = uniqueIndicesCurrent.size();
    double shookPercentage = (totalIndices > 0) ? (static_cast<double>(uniqueIndicesShook.size()) / totalIndices) * 100.0 : 0.0;
    double noshookPercentage = (totalIndices > 0) ? (static_cast<double>(uniqueIndicesNoshook.size()) / totalIndices) * 100.0 : 0.0;
    double missingPercentage = (totalIndices > 0) ? (static_cast<double>(missingIndices.size()) / totalIndices) * 100.0 : 0.0;

    // Print percentage summary
    cout << fixed << setprecision(2); // Set decimal precision to 2 places
    cout << "\nShook: " << uniqueIndicesShook.size() << " (" << shookPercentage << "%)\n";
    cout << "Noshook: " << uniqueIndicesNoshook.size() << " (" << noshookPercentage << "%)\n";
    cout << "Missing: " << missingIndices.size() << " (" << missingPercentage << "%)\n";

    cout << "\nProcessing complete." << endl;
    return 0;
}