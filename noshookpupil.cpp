#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <limits>
#include <iomanip>
#include <cmath>  // For variance calculation
#include <algorithm>  // For trimming whitespace
#include <set> // For unique indices

using namespace std;
namespace fs = filesystem;

// Function to trim whitespace from a string
string trim(const string& str) {
    string trimmed = str;
    trimmed.erase(trimmed.begin(), find_if(trimmed.begin(), trimmed.end(), [](unsigned char ch) {
        return !isspace(ch);
    }));
    trimmed.erase(find_if(trimmed.rbegin(), trimmed.rend(), [](unsigned char ch) {
        return !isspace(ch);
    }).base(), trimmed.end());
    return trimmed;
}

// Function to check if a file has a .csv extension
bool isCSVFile(const fs::path& filePath) {
    return filePath.extension() == ".csv";
}

// Function to load CSV file into a 2D vector
vector<vector<string>> loadCSV(const string& filePath) {
    vector<vector<string>> data;
    ifstream file(filePath);
    if (!file.is_open()) {
        cerr << "Error: Could not open " << filePath << endl;
        return {};
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<string> row;
        string cell;
        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        data.push_back(row);
    }

    file.close();
    return data;
}

// Function to find the column that contains "leftpupil" and "rightpupil"
pair<int, int> findPupilColumns(const vector<string>& headerRow) {
    int leftPupilCol = -1, rightPupilCol = -1;
    for (size_t i = 0; i < headerRow.size(); i++) {
        string trimmedCell = trim(headerRow[i]);
        if (trimmedCell.find("leftPupil") != string::npos) {
            leftPupilCol = i;
        }
        if (trimmedCell.find("rightPupil") != string::npos) {
            rightPupilCol = i;
        }
    }
    return {leftPupilCol, rightPupilCol};
}

// Function to find the row index of "0.2 seconds"
int findEventRow(const vector<vector<string>>& data) {
    for (size_t i = 1; i < data.size(); i++) { // Start from row 1 to skip header
        for (const string& cell : data[i]) {
            if (cell.find("0.2 seconds") != string::npos) {
                return i;  // Return the first occurrence
            }
        }
    }
    return -1;  // Not found
}

// Function to compute pupil size averages before & after estimated event time
vector<double> calculatePupilAverages(const vector<vector<string>>& data, int timeCol, int leftPupilCol, int rightPupilCol, int eventRow) {
    double sumLeftBefore = 0.0, sumRightBefore = 0.0, leftcountBefore = 0, rightcountBefore = 0;
    double sumLeftAfter = 0.0, sumRightAfter = 0.0, leftcountAfter = 0, rightcountAfter = 0;

    if (eventRow == -1) {
        return {-1, -1, -1, -1};  // No valid event row
    }

    // Estimate event start time (0.229s after "0.2 seconds" tag)
    double eventTime = stod(data[eventRow][timeCol]) + 0.229;
    double beforeTime = stod(data[eventRow][timeCol]);

    for (size_t i = 1; i < data.size(); i++) { // Start from row 1 to skip header
        if (data[i].size() <= max(leftPupilCol, rightPupilCol)) continue;

        double timeValue, leftPupilSize, rightPupilSize;
        try {
            timeValue = stod(data[i][timeCol]);
            leftPupilSize = stod(data[i][leftPupilCol]);
            rightPupilSize = stod(data[i][rightPupilCol]);
        } catch (...) {
            continue;
        }

        // Compute averages for 5 seconds before the "0.2 seconds" tag
        if (timeValue >= (beforeTime - 5.0) && timeValue <= beforeTime) {
            if (leftPupilSize > 0) {
                sumLeftBefore += leftPupilSize;
                leftcountBefore++;
            }
            if (rightPupilSize > 0) {
                sumRightBefore += rightPupilSize;
                rightcountBefore++;
            }
        }

        // Compute averages for 5 seconds after the estimated event start time
        if (timeValue >= eventTime && timeValue <= (eventTime + 5.0)) {
            if (leftPupilSize > 0) {
                sumLeftAfter += leftPupilSize;
                leftcountAfter++;
            }
            if (rightPupilSize > 0) {
                sumRightAfter += rightPupilSize;
                rightcountAfter++;
            }
        }
    }

    double avgLeftBefore = (leftcountBefore > 0) ? sumLeftBefore / leftcountBefore : -1;
    double avgRightBefore = (rightcountBefore > 0) ? sumRightBefore / rightcountBefore : -1;
    double avgLeftAfter = (leftcountAfter > 0) ? sumLeftAfter / leftcountAfter : -1;
    double avgRightAfter = (rightcountAfter > 0) ? sumRightAfter / rightcountAfter : -1;
    
    return {avgLeftBefore, avgRightBefore, avgLeftAfter, avgRightAfter};
}

int main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);

    string path = ".";
    fs::path noshookFolder = fs::path(path) / "noshook";

    cout << fixed << setprecision(3);
    cout << "Scanning CSV files in the noshook folder..." << endl;

    if (!fs::exists(noshookFolder) || !fs::is_directory(noshookFolder)) {
        cerr << "Error: 'noshook' folder does not exist!" << endl;
        return 1;
    }

    cout << "\n==== Noshook Pupil Analysis Report ====\n";
    int validCount = 0, invalidCount = 0;
    double totalLeftDiff = 0.0, totalRightDiff = 0.0;
    set<string> indicesWith02, indicesWithout02;

    for (const auto& entry : fs::directory_iterator(noshookFolder)) {
        if (fs::is_regular_file(entry.path()) && isCSVFile(entry.path())) {
            string fileName = entry.path().filename().string();
            string fileIndex = fileName.substr(0, 5);

            vector<vector<string>> data = loadCSV(entry.path().string());
            if (data.empty()) {
                cout << "Index " << fileIndex << " -> ERROR: Could not load CSV ❌" << endl;
                invalidCount++;
                continue;
            }

            pair<int, int> pupilColumns = findPupilColumns(data[0]);
            if (pupilColumns.first == -1 || pupilColumns.second == -1) {
                cout << "Index " << fileIndex << " -> ERROR: 'leftpupil' or 'rightpupil' column not found ❌" << endl;
                invalidCount++;
                continue;
            }

            int eventRow = findEventRow(data);
            if (eventRow == -1) {
                cout << "Index " << fileIndex << " -> ❌ NO '0.2 seconds' keyword found ❌" << endl;
                indicesWithout02.insert(fileIndex);
                continue;
            }

            vector<double> pupilAverages = calculatePupilAverages(data, 0, pupilColumns.first, pupilColumns.second, eventRow);
            double leftDiff = pupilAverages[2] - pupilAverages[0];
            double rightDiff = pupilAverages[3] - pupilAverages[1];

            totalLeftDiff += leftDiff;
            totalRightDiff += rightDiff;
            validCount++;

            cout << "Index " << fileIndex << " -> Left Diff: " << leftDiff << ", Right Diff: " << rightDiff << " ✅" << endl;
        }
    }

    cout << "\n==== Final Summary ====\n";
    cout << "Valid Data: " << validCount << ", Invalid Data: " << invalidCount << endl;
    cout << "Avg Left Diff: " << (validCount ? totalLeftDiff / validCount : 0) << endl;
    cout << "Avg Right Diff: " << (validCount ? totalRightDiff / validCount : 0) << endl;

    return 0;
}