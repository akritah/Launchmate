const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

export interface User {
  id: string;
  email: string;
  full_name: string;
  role: string;
  department: string;
  location: string;
  is_admin: boolean;
  phone_number?: string;
  organization_id?: string;
}

export interface Task {
  id: string;
  user_id: string;
  title: string;
  description?: string;
  category: string;
  due_date?: string;
  priority: string;
  required: boolean;
  status: "pending" | "in_progress" | "completed";
  completed_at?: string;
  is_custom?: boolean;
}

export interface TaskStats {
  total_tasks: number;
  completed_tasks: number;
  in_progress_tasks: number;
  pending_tasks: number;
  completion_percentage: number;
  overdue_tasks: number;
}

export interface Citation {
  title: string;
  section?: string;
  source_file?: string;
  snippet: string;
  score: number;
}

export interface ChatMessage {
  id?: string;
  conversation_id?: string;
  sender: "user" | "assistant";
  content: string;
  category?: string;
  confidence?: number;
  citations?: Citation[];
  agent_action?: any;
  execution_steps?: string[];
  created_at?: string;
}

export interface Escalation {
  id: string;
  user_id: string;
  user_name?: string;
  user_email?: string;
  reason: string;
  trigger_message: string;
  sensitivity_reason?: string;
  priority: string;
  status: "open" | "in_progress" | "resolved";
  created_at: string;
  resolved_at?: string;
  resolution_notes?: string;
  hr_assigned_to?: string;
}

export interface ChecklistTemplate {
  id: string;
  role: string;
  location: string;
  department: string;
  title: string;
  description?: string;
  category: string;
  due_days_from_hire: number;
  priority: string;
  required: boolean;
  template_type?: string;
}

export interface DocumentItem {
  id: string;
  title: string;
  category: string;
  source_file: string;
  file_type?: string;
  file_size?: number;
  chunk_count: number;
  status: string;
  created_at: string;
}

function getHeaders(token?: string): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  return headers;
}

export const api = {
  // ─── Auth ───
  async getDemoUsers(): Promise<User[]> {
    const res = await fetch(`${API_BASE_URL}/auth/demo-users`);
    if (!res.ok) throw new Error("Failed to fetch demo users");
    return res.json();
  },

  async quickSwitch(userId: string): Promise<{ access_token: string; user: User }> {
    const res = await fetch(`${API_BASE_URL}/auth/quick-switch`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({ user_id: userId }),
    });
    if (!res.ok) throw new Error("Quick switch failed");
    return res.json();
  },

  async login(email: string, password: string): Promise<{ access_token: string; user: User }> {
    const res = await fetch(`${API_BASE_URL}/auth/login`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Login failed" }));
      throw new Error(err.detail || "Invalid credentials");
    }
    return res.json();
  },

  async signup(data: any): Promise<{ access_token: string; user: User }> {
    const res = await fetch(`${API_BASE_URL}/auth/signup`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Signup failed" }));
      throw new Error(err.detail || "Signup failed");
    }
    return res.json();
  },

  async oauthLogin(provider: string, email: string, full_name: string, role?: string, department?: string, location?: string): Promise<{ access_token: string; user: User }> {
    const res = await fetch(`${API_BASE_URL}/auth/oauth`, {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({ provider, email, full_name, role, department, location }),
    });
    if (!res.ok) throw new Error("OAuth login failed");
    return res.json();
  },

  // ─── Tasks ───
  async getTasks(token: string): Promise<Task[]> {
    const res = await fetch(`${API_BASE_URL}/tasks`, {
      headers: getHeaders(token),
    });
    if (!res.ok) throw new Error("Failed to fetch tasks");
    return res.json();
  },

  async updateTaskStatus(token: string, taskId: string, status: string): Promise<Task> {
    const res = await fetch(`${API_BASE_URL}/tasks/${taskId}/status`, {
      method: "PATCH",
      headers: getHeaders(token),
      body: JSON.stringify({ status }),
    });
    if (!res.ok) throw new Error("Failed to update task status");
    return res.json();
  },

  async updateTask(token: string, taskId: string, data: Partial<Task>): Promise<Task> {
    const res = await fetch(`${API_BASE_URL}/tasks/${taskId}`, {
      method: "PATCH",
      headers: getHeaders(token),
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Failed to update task");
    return res.json();
  },

  async deleteTask(token: string, taskId: string): Promise<void> {
    const res = await fetch(`${API_BASE_URL}/tasks/${taskId}`, {
      method: "DELETE",
      headers: getHeaders(token),
    });
    if (!res.ok) throw new Error("Failed to delete task");
  },

  async createTask(token: string, data: any): Promise<Task> {
    const res = await fetch(`${API_BASE_URL}/tasks`, {
      method: "POST",
      headers: getHeaders(token),
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Failed to create task");
    return res.json();
  },

  async getTaskStats(token: string): Promise<TaskStats> {
    const res = await fetch(`${API_BASE_URL}/tasks/stats`, {
      headers: getHeaders(token),
    });
    if (!res.ok) throw new Error("Failed to fetch stats");
    return res.json();
  },

  // ─── Chat & LangGraph ───
  async getChatHistory(token: string, conversationId: string = "default"): Promise<ChatMessage[]> {
    const res = await fetch(`${API_BASE_URL}/chat/history?conversation_id=${encodeURIComponent(conversationId)}`, {
      headers: getHeaders(token),
    });
    if (!res.ok) return [];
    return res.json();
  },

  async clearChatHistory(token: string, conversationId: string = "default"): Promise<void> {
    await fetch(`${API_BASE_URL}/chat/history?conversation_id=${encodeURIComponent(conversationId)}`, {
      method: "DELETE",
      headers: getHeaders(token),
    });
  },

  async getGraphMermaid(): Promise<{ mermaid: string; nodes: string[]; description: string }> {
    const res = await fetch(`${API_BASE_URL}/chat/graph/mermaid`);
    if (!res.ok) throw new Error("Failed to fetch LangGraph mermaid diagram");
    return res.json();
  },

  async chatGraph(token: string, content: string, conversationId: string = "default"): Promise<ChatMessage> {
    const res = await fetch(`${API_BASE_URL}/chat/graph`, {
      method: "POST",
      headers: getHeaders(token),
      body: JSON.stringify({ content, conversation_id: conversationId, engine: "langgraph" }),
    });
    if (!res.ok) throw new Error("LangGraph chat failed");
    return res.json();
  },

  streamChat(
    token: string,
    message: string,
    conversationId: string = "default",
    onMessage: (data: any) => void,
    onError: (err: any) => void,
    onDone: () => void
  ) {
    const url = `${API_BASE_URL}/chat/stream?message=${encodeURIComponent(message)}&conversation_id=${encodeURIComponent(conversationId)}&token=${encodeURIComponent(token)}`;
    const eventSource = new EventSource(url);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "done") {
          eventSource.close();
          onDone();
        } else {
          onMessage(data);
        }
      } catch (e) {
        console.error("Error parsing SSE data", e);
      }
    };

    eventSource.onerror = (err) => {
      eventSource.close();
      onError(err);
    };

    return () => eventSource.close();
  },

  async escalateIssue(token: string, data: any): Promise<any> {
    const res = await fetch(`${API_BASE_URL}/chat/escalate`, {
      method: "POST",
      headers: getHeaders(token),
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Escalation failed");
    return res.json();
  },

  // ─── Admin Portal ───
  async getNewJoiners(token: string): Promise<User[]> {
    const res = await fetch(`${API_BASE_URL}/admin/joiners`, {
      headers: getHeaders(token),
    });
    if (!res.ok) throw new Error("Failed to fetch joiners");
    return res.json();
  },

  async getJoinerTasks(token: string, userId: string): Promise<Task[]> {
    const res = await fetch(`${API_BASE_URL}/admin/joiners/${userId}/tasks`, {
      headers: getHeaders(token),
    });
    if (!res.ok) throw new Error("Failed to fetch joiner tasks");
    return res.json();
  },

  async createJoiner(token: string, data: any): Promise<User> {
    const res = await fetch(`${API_BASE_URL}/admin/joiners`, {
      method: "POST",
      headers: getHeaders(token),
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Failed to create joiner");
    return res.json();
  },

  async assignTask(token: string, data: any): Promise<Task[]> {
    const res = await fetch(`${API_BASE_URL}/admin/assign-task`, {
      method: "POST",
      headers: getHeaders(token),
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Failed to assign task");
    return res.json();
  },

  async updateAdminTaskStatus(token: string, taskId: string, status: string): Promise<Task> {
    const res = await fetch(`${API_BASE_URL}/admin/tasks/${taskId}/status`, {
      method: "PATCH",
      headers: getHeaders(token),
      body: JSON.stringify({ status }),
    });
    if (!res.ok) throw new Error("Failed to update status");
    return res.json();
  },

  async getOverdueTasks(token: string): Promise<Task[]> {
    const res = await fetch(`${API_BASE_URL}/admin/overdue-tasks`, {
      headers: getHeaders(token),
    });
    if (!res.ok) throw new Error("Failed to fetch overdue tasks");
    return res.json();
  },

  async notifyDeadlineOverdue(token: string, taskId: string): Promise<any> {
    const res = await fetch(`${API_BASE_URL}/admin/notify-deadline-overdue?task_id=${encodeURIComponent(taskId)}`, {
      method: "POST",
      headers: getHeaders(token),
    });
    if (!res.ok) throw new Error("Failed to notify deadline");
    return res.json();
  },

  async getTemplates(token: string): Promise<ChecklistTemplate[]> {
    const res = await fetch(`${API_BASE_URL}/admin/templates`, {
      headers: getHeaders(token),
    });
    if (!res.ok) throw new Error("Failed to fetch templates");
    return res.json();
  },

  async createTemplate(token: string, data: any): Promise<ChecklistTemplate> {
    const res = await fetch(`${API_BASE_URL}/admin/templates`, {
      method: "POST",
      headers: getHeaders(token),
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Failed to create template");
    return res.json();
  },

  async deleteTemplate(token: string, templateId: string): Promise<void> {
    const res = await fetch(`${API_BASE_URL}/admin/templates/${templateId}`, {
      method: "DELETE",
      headers: getHeaders(token),
    });
    if (!res.ok) throw new Error("Failed to delete template");
  },

  async getEscalations(token: string): Promise<Escalation[]> {
    const res = await fetch(`${API_BASE_URL}/admin/escalations`, {
      headers: getHeaders(token),
    });
    if (!res.ok) throw new Error("Failed to fetch escalations");
    return res.json();
  },

  async resolveEscalation(token: string, escId: string, resolutionNotes: string, hrAssignedTo?: string): Promise<any> {
    const res = await fetch(`${API_BASE_URL}/admin/escalations/${escId}/resolve`, {
      method: "POST",
      headers: getHeaders(token),
      body: JSON.stringify({ resolution_notes: resolutionNotes, hr_assigned_to: hrAssignedTo }),
    });
    if (!res.ok) throw new Error("Failed to resolve escalation");
    return res.json();
  },

  async triggerNudges(token: string, simulate: boolean = true): Promise<any> {
    const res = await fetch(`${API_BASE_URL}/admin/nudges/trigger?simulate=${simulate}`, {
      method: "POST",
      headers: getHeaders(token),
    });
    if (!res.ok) throw new Error("Failed to trigger nudges");
    return res.json();
  },

  async getDocuments(token: string): Promise<DocumentItem[]> {
    const res = await fetch(`${API_BASE_URL}/admin/documents`, {
      headers: getHeaders(token),
    });
    if (!res.ok) throw new Error("Failed to fetch documents");
    return res.json();
  },

  async uploadDocument(token: string, data: any): Promise<DocumentItem> {
    const res = await fetch(`${API_BASE_URL}/admin/documents`, {
      method: "POST",
      headers: getHeaders(token),
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Failed to upload document");
    return res.json();
  },

  async uploadDocumentFile(token: string, file: File, category: string): Promise<DocumentItem> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("category", category);
    const res = await fetch(`${API_BASE_URL}/admin/documents/upload-file`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: formData,
    });
    if (!res.ok) throw new Error("Failed to upload document file");
    return res.json();
  },

  async deleteDocument(token: string, docId: string): Promise<void> {
    const res = await fetch(`${API_BASE_URL}/admin/documents/${docId}`, {
      method: "DELETE",
      headers: getHeaders(token),
    });
    if (!res.ok) throw new Error("Failed to delete document");
  },

  async getCommonQuestions(token: string): Promise<any[]> {
    const res = await fetch(`${API_BASE_URL}/admin/common-questions`, {
      headers: getHeaders(token),
    });
    if (!res.ok) return [];
    return res.json();
  },

  async getSystemStats(token: string): Promise<any> {
    const res = await fetch(`${API_BASE_URL}/admin/stats`, {
      headers: getHeaders(token),
    });
    if (!res.ok) throw new Error("Failed to fetch system stats");
    return res.json();
  },

  // ─── Live Support / Call Routing ───
  async getAgentLines(token?: string, topic?: string): Promise<any> {
    const query = topic ? `?topic=${encodeURIComponent(topic)}` : "";
    const res = await fetch(`${API_BASE_URL}/support/lines${query}`, {
      headers: getHeaders(token),
    });
    if (!res.ok) throw new Error("Failed to fetch agent lines");
    return res.json();
  },

  async requestAgentCall(token: string | undefined, data: any): Promise<any> {
    const res = await fetch(`${API_BASE_URL}/support/request-call`, {
      method: "POST",
      headers: getHeaders(token),
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Failed to request agent call");
    return res.json();
  },

  async getCallHistory(token: string): Promise<any[]> {
    const res = await fetch(`${API_BASE_URL}/support/calls`, {
      headers: getHeaders(token),
    });
    if (!res.ok) return [];
    return res.json();
  },
};
