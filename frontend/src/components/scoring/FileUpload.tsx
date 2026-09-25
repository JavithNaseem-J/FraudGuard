import React, { useRef, useState } from 'react';
import { UploadCloud, FileText } from 'lucide-react';
import { Button } from '@/components/ui';
import { cn } from '@/utils/cn';

export interface FileUploadProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onFileSelected, disabled = false }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndPassFile = (file: File) => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext !== 'csv' && ext !== 'json') {
      setErrorMessage('Unsupported file format. Please upload a .csv or .json file.');
      return;
    }
    setErrorMessage(null);
    onFileSelected(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (disabled) return;
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      validateAndPassFile(file);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      validateAndPassFile(file);
      // reset input value so re-selecting same file triggers change
      e.target.value = '';
    }
  };

  return (
    <div className="w-full space-y-2">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={cn(
          'w-full min-h-[220px] rounded-lg border-2 border-dashed flex flex-col items-center justify-center p-6 text-center transition-colors',
          isDragOver
            ? 'border-primary bg-blue-50/40'
            : 'border-slate-border bg-white hover:border-slate-400',
          disabled && 'opacity-60 cursor-not-allowed pointer-events-none'
        )}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.json"
          onChange={handleFileChange}
          className="hidden"
          disabled={disabled}
        />

        <div className="w-12 h-12 rounded-full bg-blue-50 border border-blue-100 flex items-center justify-center text-primary mb-3">
          <UploadCloud className="w-6 h-6 stroke-[1.75]" />
        </div>

        <p className="text-sm font-semibold text-slate-main mb-1">
          Drag and drop your CSV or JSON file here
        </p>
        <p className="text-xs text-slate-secondary mb-4">
          Supports .csv, .json. Large CSV uploads load the first scoring batch.
        </p>

        <Button
          type="button"
          variant="primary"
          size="sm"
          disabled={disabled}
          onClick={() => fileInputRef.current?.click()}
          icon={<FileText className="w-3.5 h-3.5" />}
        >
          Choose File
        </Button>
      </div>

      {errorMessage && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-input text-xs text-status-danger font-medium">
          {errorMessage}
        </div>
      )}
    </div>
  );
};

export default FileUpload;
