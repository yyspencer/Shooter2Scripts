#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cctype>
#include <cmath>

using namespace std;
namespace fs = filesystem;

// ---------- helpers ---------------------------------------------------------

string toLower(string s) {
    transform(s.begin(), s.end(), s.begin(),
              [](unsigned char c) { return tolower(c); });
    return s;
}

string extractIndex(const string& name) {
    return name.substr(0, 5); // first 5 chars
}

bool parseDouble(const string& s, double& out) {
    try {
        size_t pos;
        out = stod(s, &pos);
        return pos == s.size();
    } catch (...) {
        return false;
    }
}

struct ColSet {
    int px{-1}, py{-1}, pz{-1};
    int rx{-1}, ry{-1}, rz{-1};
    int roomEvent{-1};
    bool valid() const {
        return px >= 0 && py >= 0 && pz >= 0 &&
               rx >= 0 && ry >= 0 && rz >= 0 &&
               roomEvent >= 0;
    }
};

ColSet locateColumns(const string& header) {
    stringstream ss(header);
    string cell;
    int idx = 0;
    ColSet cs;

    while (getline(ss, cell, ',')) {
        string lc = toLower(cell);
        if (lc.find("playervr.x") != string::npos) cs.px = idx;
        if (lc.find("playervr.y") != string::npos) cs.py = idx;
        if (lc.find("playervr.z") != string::npos) cs.pz = idx;
        if (lc.find("robot.x")    != string::npos) cs.rx = idx;
        if (lc.find("robot.y")    != string::npos) cs.ry = idx;
        if (lc.find("robot.z")    != string::npos) cs.rz = idx;
        if (lc.find("roomevent")  != string::npos) cs.roomEvent = idx;
        ++idx;
    }
    return cs;
}

// ---------- main ------------------------------------------------------------

int main() {
    const fs::path surveyDir = "survey";
    const fs::path speedDir = "surveyspeed";

    if (!fs::exists(surveyDir) || !fs::is_directory(surveyDir)) {
        cerr << "Error: 'survey' folder not found.\n";
        return 1;
    }
    if (!fs::exists(speedDir)) fs::create_directory(speedDir);

    map<string, vector<string>> speedLines;

    for (const auto& entry : fs::directory_iterator(surveyDir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".csv") continue;

        string index = extractIndex(entry.path().filename().string());
        ifstream fin(entry.path());
        if (!fin) {
            cerr << "Cannot open " << entry.path() << '\n';
            continue;
        }

        string header;
        if (!getline(fin, header)) {
            cerr << "Empty file " << entry.path() << '\n';
            continue;
        }

        ColSet cols = locateColumns(header);
        if (!cols.valid()) {
            cerr << "Missing required columns in " << entry.path() << '\n';
            continue;
        }

        bool firstRow = true;
        double pPrev[3]{}, rPrev[3]{};
        string line;

        auto& buf = speedLines[index];
        buf.emplace_back("playerSpeed robotSpeed");

        while (getline(fin, line)) {
            stringstream ss(line);
            vector<string> cells;
            string cell;
            while (getline(ss, cell, ',')) cells.push_back(cell);

            // Check roomEvent column
            if (cells.size() <= cols.roomEvent) break;

            string eventText = toLower(cells[cols.roomEvent]);
            if (eventText.find("robot entered survey room") != string::npos) {
                break; // stop before this line
            }

            if (firstRow) {
                buf.emplace_back("0 0");
                parseDouble(cells[cols.px], pPrev[0]);
                parseDouble(cells[cols.py], pPrev[1]);
                parseDouble(cells[cols.pz], pPrev[2]);
                parseDouble(cells[cols.rx], rPrev[0]);
                parseDouble(cells[cols.ry], rPrev[1]);
                parseDouble(cells[cols.rz], rPrev[2]);
                firstRow = false;
                continue;
            }

            double pCur[3], rCur[3];
            bool ok =
                parseDouble(cells[cols.px], pCur[0]) &&
                parseDouble(cells[cols.py], pCur[1]) &&
                parseDouble(cells[cols.pz], pCur[2]) &&
                parseDouble(cells[cols.rx], rCur[0]) &&
                parseDouble(cells[cols.ry], rCur[1]) &&
                parseDouble(cells[cols.rz], rCur[2]);

            if (!ok) {
                buf.emplace_back("-1");
                continue;
            }

            double pSpeed = sqrt(pow(pCur[0] - pPrev[0], 2) +
                                 pow(pCur[1] - pPrev[1], 2) +
                                 pow(pCur[2] - pPrev[2], 2));

            double rSpeed = sqrt(pow(rCur[0] - rPrev[0], 2) +
                                 pow(rCur[1] - rPrev[1], 2) +
                                 pow(rCur[2] - rPrev[2], 2));

            ostringstream oss;
            oss << pSpeed << ' ' << rSpeed;
            buf.emplace_back(oss.str());

            copy(begin(pCur), end(pCur), begin(pPrev));
            copy(begin(rCur), end(rCur), begin(rPrev));
        }
    }

    // Write to speed folder
    for (const auto& [idx, lines] : speedLines) {
        fs::path outPath = speedDir / (idx + ".txt");
        ofstream fout(outPath);
        if (!fout) {
            cerr << "Cannot write to " << outPath << '\n';
            continue;
        }
        for (const auto& ln : lines) fout << ln << '\n';
        cout << "Wrote " << outPath << '\n';
    }

    cout << "Done.\n";
    return 0;
}