import React from 'react';
import { CheckCircle2, AlertTriangle } from 'lucide-react';
import { Badge } from '@/components/ui';

export interface SchemaValidationResult {
  isValid: boolean;
  presentFields: string[];
  missingFields: string[];
  unexpectedFields: string[];
}

export interface SchemaValidationProps {
  validationResult: SchemaValidationResult;
}

export const SchemaValidation: React.FC<SchemaValidationProps> = ({ validationResult }) => {
  const { isValid, missingFields, presentFields } = validationResult;

  if (isValid) {
    return (
      <div className="flex items-center gap-2 text-xs text-status-success font-medium">
        <CheckCircle2 className="w-4 h-4 text-status-success shrink-0" />
        <span>Schema Valid ({presentFields.length} fields verified)</span>
      </div>
    );
  }

  return (
    <div className="p-3 bg-red-50/70 border border-red-200 rounded-lg text-xs space-y-2">
      <div className="flex items-center gap-2 text-status-danger font-semibold">
        <AlertTriangle className="w-4 h-4 shrink-0" />
        <span>Schema Invalid — Missing Required Transaction Fields</span>
      </div>

      <div className="text-slate-secondary text-[11px]">
        The following required features were not found in the uploaded columns:
      </div>

      <div className="flex flex-wrap gap-1.5 pt-1">
        {missingFields.map((field) => (
          <Badge key={field} variant="danger" size="sm">
            {field}
          </Badge>
        ))}
      </div>
    </div>
  );
};

export default SchemaValidation;
