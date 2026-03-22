#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

function readInt32LE(buffer, offset) {
    return buffer.readInt32LE(offset);
}

function readFloat(buffer, offset) {
    return buffer.readFloatLE(offset);
}

function parseDayFile(filepath) {
    const buffer = fs.readFileSync(filepath);
    const records = [];
    const recordSize = 32;

    for (let i = 0; i + recordSize <= buffer.length; i += recordSize) {
        const date = readInt32LE(buffer, i);
        const open = readInt32LE(buffer, i + 4) / 100;
        const high = readInt32LE(buffer, i + 8) / 100;
        const low = readInt32LE(buffer, i + 12) / 100;
        const close = readInt32LE(buffer, i + 16) / 100;
        const amount = readFloat(buffer, i + 20);
        const volume = readInt32LE(buffer, i + 24);

        records.push({
            date: date.toString(),
            open, high, low, close,
            amount, volume
        });
    }

    return records;
}

function toCSV(data, filepath) {
    if (data.length === 0) return;

    const headers = Object.keys(data[0]).join(',');
    const rows = data.map(r => Object.values(r).join(','));
    const csv = [headers, ...rows].join('\n');

    fs.writeFileSync(filepath, csv, 'utf8');
    return filepath;
}

const inputFile = 'D:/new_tdx/vipdoc/sz/lday/sz301062.day';
const outputFile = 'D:/新建文件夹/claude/test_output/sz301062.day.csv';

console.log('Parsing TDX day file...');
console.log('Input:', inputFile);
const data = parseDayFile(inputFile);
console.log(`Parsed ${data.length} records`);
console.log('First 3 records:');
data.slice(0, 3).forEach(r => {
    console.log(`  ${r.date} O:${r.open} H:${r.high} L:${r.low} C:${r.close} V:${r.volume}`);
});
console.log('Last 3 records:');
data.slice(-3).forEach(r => {
    console.log(`  ${r.date} O:${r.open} H:${r.high} L:${r.low} C:${r.close} V:${r.volume}`);
});

toCSV(data, outputFile);
console.log('Output:', outputFile);
console.log('Done!');
